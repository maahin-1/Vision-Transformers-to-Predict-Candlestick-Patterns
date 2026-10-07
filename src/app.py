"""Candlestick pattern tracker: live chart + ViT prediction in one page.

    uv run src/app.py

Type a ticker, press Track. In Live mode the newest closed 1-minute candles are polled from
Yahoo Finance; in Replay mode the latest session is replayed one candle per second (useful
when the market is closed). The ViT reads a picture rendered from the same candles, not your
screen, so it works wherever the window sits.
"""
import base64
import io
import threading
import time
import uuid

import plotly.graph_objects as go
from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from PIL import Image

import alerts
import context
import history
import market
import predictor
import render
import rules
import scanner
import sessions

CLASSES = rules.CLASSES
COLORS = ['#ebdc9c', '#8ccfa6', '#c1abec', '#e8a3ca', '#ff8080']
LIVE_POLL_MS = 30_000
SLOW_POLL_MS = 60_000  # 5m and longer candles change less often
REPLAY_MS = 1_000
MODEL_WINDOW = 8  # candles the ViT crop covers
START_TICKER = 'AMZN'

HISTORY = history.HistoryLog()
FEED = alerts.AlertFeed()
_telegram = {'error': ''}  # last Telegram failure, shown in the alerts card
_lock = threading.Lock()  # guards S and HISTORY
_infer_lock = threading.Lock()  # one forward pass at a time
def _new_state():
    return {'ticker': None, 'mode': 'live', 'timeframe': market.DEFAULT_TIMEFRAME, 'df': None, 'cursor': 0,
            'fetched': 0.0, 'warning': '', 'error': '', 'cache': (None, None), 'used': time.monotonic()}


S = sessions.SessionState(_new_state)  # one state per browser tab

app = Dash(__name__, title='Candlestick Pattern Tracker')

PAGE = {'fontFamily': 'system-ui, Segoe UI, sans-serif', 'maxWidth': '1280px', 'margin': '0 auto', 'padding': '16px',
        'color': '#1f2937'}
CARD = {'background': '#fff', 'border': '1px solid #e5e7eb', 'borderRadius': '10px', 'padding': '14px'}

def _build_layout():
    return html.Div(style=PAGE, children=[
        html.H2('Candlestick Pattern Tracker', style={'margin': '0 0 12px'}),
        html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
            dcc.Input(id='ticker-input', type='text', value=START_TICKER, placeholder='Ticker, e.g. AAPL',
                      debounce=False, n_submit=0, style={'padding': '8px 10px', 'width': '180px', 'fontSize': '16px'}),
            dcc.Dropdown(id='timeframe', value=market.DEFAULT_TIMEFRAME, clearable=False, searchable=False,
                         options=[{'label': tf, 'value': tf} for tf in market.TIMEFRAMES],
                         style={'width': '80px', 'fontSize': '16px'}),
            html.Button('Track', id='track-btn', n_clicks=0,
                        style={'padding': '8px 18px', 'fontSize': '16px', 'cursor': 'pointer'}),
            dcc.RadioItems(id='mode', value='live', inline=True, inputStyle={'marginRight': '4px', 'marginLeft': '12px'},
                           options=[{'label': 'Live (polls Yahoo)', 'value': 'live'},
                                    {'label': 'Replay (1 candle/s)', 'value': 'replay'}]),
        ]),
        html.Div(id='status', style={'margin': '10px 0', 'fontSize': '14px'}),
        html.Div(style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}, children=[
            html.Div(style={**CARD, 'flex': '3 1 640px', 'minWidth': '0'},
                     children=dcc.Graph(id='chart', config={'displayModeBar': False}, style={'height': '440px'})),
            html.Div(style={**CARD, 'flex': '1 1 320px'}, children=[
                html.Div('ViT prediction', style={'fontWeight': 600, 'marginBottom': '8px'}),
                html.Div(id='prediction'),
                html.Div(id='rule-check', style={'marginTop': '12px', 'fontSize': '14px'}),
                html.Div(id='context', style={'marginTop': '6px', 'fontSize': '14px'}),
                html.Div('What the model sees', style={'fontWeight': 600, 'margin': '14px 0 6px'}),
                html.Img(id='model-view', style={'imageRendering': 'pixelated', 'border': '1px solid #e5e7eb'}),
            ]),
        ]),
        html.Div(style={**CARD, 'marginTop': '14px', 'overflowX': 'auto'}, children=[
            html.Div('Watchlist scanner', style={'fontWeight': 600, 'marginBottom': '8px'}),
            html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                dcc.Input(id='scan-input', type='text', value='AAPL, MSFT, TSLA, NVDA, AMZN',
                          placeholder=f'Up to {scanner.MAX_TICKERS} tickers, comma separated',
                          style={'padding': '6px 10px', 'width': '380px', 'fontSize': '15px'}),
                html.Button('Scan', id='scan-btn', n_clicks=0, style={'padding': '6px 16px', 'cursor': 'pointer'}),
                dcc.Checklist(id='scan-auto', value=[], options=[{'label': ' Auto-refresh', 'value': 'auto'}]),
                dcc.Checklist(id='scan-fit', value=[], options=[{'label': ' Only signals that fit the trend',
                                                                 'value': 'fit'}]),
                html.Span(id='scan-status', style={'fontSize': '13px', 'color': '#6b7280'}),
            ]),
            html.Div(id='scan-results', style={'marginTop': '10px'}),
        ]),
        html.Div(style={**CARD, 'marginTop': '14px', 'overflowX': 'auto'}, children=[
            html.Div('Alerts', style={'fontWeight': 600, 'marginBottom': '4px'}),
            html.Div('Fires for live signals only (never Replay or a closed market), once per candle, while this page '
                     'is open. Applies to the main view in Live mode and to scanner runs.',
                     style={'fontSize': '12px', 'color': '#6b7280', 'marginBottom': '8px'}),
            html.Div(style={'display': 'flex', 'gap': '14px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
                html.Span('Min confidence %'),
                dcc.Input(id='alert-conf', type='number', min=50, max=99, step=1, value=85,
                          style={'width': '70px', 'padding': '4px 6px'}),
                dcc.Checklist(id='alert-opts', value=['agree'], inline=True,
                              inputStyle={'marginRight': '4px', 'marginLeft': '10px'},
                              options=[{'label': 'rules must agree', 'value': 'agree'},
                                       {'label': 'must fit the trend', 'value': 'fit'},
                                       {'label': 'browser notification', 'value': 'browser'},
                                       {'label': 'Telegram', 'value': 'telegram'}]),
            ]),
            html.Div(id='telegram-status', style={'fontSize': '12px', 'color': '#6b7280', 'marginTop': '6px'}),
            html.Div(id='alerts-feed', style={'marginTop': '8px'}),
        ]),
        html.Div(style={**CARD, 'marginTop': '14px', 'overflowX': 'auto'}, children=[
            html.Div(f'Last {MODEL_WINDOW} candles (the ones the model reads) - compare with Yahoo or your broker',
                     style={'fontWeight': 600, 'marginBottom': '8px'}),
            html.Div(id='table'),
        ]),
        html.Div(style={**CARD, 'marginTop': '14px', 'overflowX': 'auto'}, children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                            'marginBottom': '8px'}, children=[
                html.Div('Detection history (logged to logs/detections.csv)', style={'fontWeight': 600}),
                html.Button('Download CSV', id='export-btn', n_clicks=0, style={'padding': '4px 12px', 'cursor': 'pointer'}),
                dcc.Download(id='export-download'),
            ]),
            html.Div(id='history'),
        ]),
        html.Div('Educational demo, not financial advice. The model has no "no pattern" class: it always picks one of '
                 'five, so read the confidence and the rule check.',
                 style={'marginTop': '12px', 'fontSize': '12px', 'color': '#6b7280'}),
        dcc.Interval(id='tick', interval=REPLAY_MS, n_intervals=0),
        dcc.Interval(id='scan-tick', interval=LIVE_POLL_MS, n_intervals=0, disabled=True),
        dcc.Store(id='open-ticker'),
        dcc.Store(id='sid', data=uuid.uuid4().hex),
        dcc.Store(id='alert-signal'), dcc.Store(id='alert-sink-1'), dcc.Store(id='alert-sink-2'),
        dcc.Interval(id='alert-tick', interval=5_000, n_intervals=0),
    ])


app.layout = _build_layout  # called per page load


def _infer(image):
    with _infer_lock:
        return predictor.predict(image)


def _alert_settings(conf, opts):
    opts = opts or []
    pct = min(99, max(50, conf if isinstance(conf, (int, float)) else 85))
    return alerts.Settings(min_confidence=pct / 100, require_agree='agree' in opts, require_fit='fit' in opts)


def _notify(rows, conf, opts):
    """Add matching rows to the alert feed and, if enabled, send each new one to Telegram."""
    settings = _alert_settings(conf, opts)
    creds = alerts.telegram_credentials() if 'telegram' in (opts or []) else None
    for row in rows:
        if not alerts.matches(row, settings):
            continue
        message = FEED.add(row)
        if message and creds:
            threading.Thread(target=_send_telegram, args=(message, creds), daemon=True).start()


def _send_telegram(message, creds):
    ok, error = alerts.send_telegram(message, *creds)
    _telegram['error'] = '' if ok else error


def _poll_ms(timeframe):
    return LIVE_POLL_MS if timeframe in ('1m',) else SLOW_POLL_MS


def _start(ticker_text, mode, timeframe):
    """(Re)load the ticker. Raises MarketDataError with a user-readable message."""
    ticker = market.normalize_ticker(ticker_text)
    df = market.fetch_candles(ticker, timeframe)
    S.update(ticker=ticker, mode=mode, timeframe=timeframe, df=df, cursor=market.MIN_CANDLES, fetched=time.monotonic(), warning='',
             cache=(None, None))


def _visible():
    df = S['df']
    if S['mode'] == 'replay':
        return df.iloc[:S['cursor']]
    return market.closed_only(df, market.timeframe_seconds(S['timeframe']))


def _status(visible):
    last = visible['Datetime'].iloc[-1]
    tf = S['timeframe']
    state, age = market.market_status(S['df'], market.timeframe_seconds(tf))
    stamp = f"{last:%Y-%m-%d} " + market.format_time(last, tf) if tf != '1d' else market.format_time(last, tf)
    if S['mode'] == 'replay':
        text = f"REPLAY {S['ticker']} {tf} - candle {len(visible)}/{len(S['df'])} - {stamp}"
        color = '#2563eb'
    elif state == 'live':
        text = f"LIVE {S['ticker']} {tf} - last closed candle {stamp} ({age:.0f} min ago; Yahoo may be delayed)"
        color = '#059669'
    else:
        text = (f"MARKET CLOSED {S['ticker']} {tf} - latest candle {stamp}. "
                f'Switch to Replay to watch it play out.')
        color = '#b45309'
    children = [html.Span(text, style={'color': color, 'fontWeight': 600})]
    if S['warning']:
        children.append(html.Span(f"  |  {S['warning']}", style={'color': '#b91c1c'}))
    return children


def _figure(visible):
    data = visible.iloc[-render.N_CANDLES:]
    labels = [market.format_time(t, S['timeframe']) for t in data['Datetime']]
    fig = go.Figure(go.Candlestick(x=labels, open=data['Open'], high=data['High'], low=data['Low'],
                                   close=data['Close'], increasing_line_color='#3D9970', decreasing_line_color='#FF4136'))
    n = len(labels)  # category axes put candle i at position i
    fig.add_vrect(x0=n - MODEL_WINDOW - 0.5, x1=n - 0.5, fillcolor='#6366f1', opacity=0.10, line_width=0,
                  annotation_text='model window', annotation_position='top left', annotation_font_size=11)
    fig.update_layout(margin=dict(l=50, r=20, t=20, b=40), xaxis_rangeslider_visible=False,
                      uirevision=f"{S['ticker']}-{S['timeframe']}", height=420, template='plotly_white',
                      showlegend=False, xaxis_type='category', xaxis_nticks=8)
    return fig


def _predict(visible):
    key = (S['ticker'], S['timeframe'], visible['Datetime'].iloc[-1])
    if S['cache'][0] == key:
        return S['cache'][1]
    data = visible.iloc[-render.N_CANDLES:]
    image = render.render_chart(data['Open'], data['High'], data['Low'], data['Close'], render.training_style_times())
    probs = _infer(image)
    view = Image.fromarray(predictor.model_view(image)).resize((216, 312), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    view.save(buf, format='PNG')
    src = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
    S['cache'] = (key, (probs, src))
    return probs, src


def _prediction_panel(probs):
    top = max(range(len(probs)), key=probs.__getitem__)
    weak = probs[top] < 0.5
    rows = [html.Div(f'{CLASSES[top]}  {probs[top]:.1%}' + ('  (weak)' if weak else ''),
                     style={'background': COLORS[top], 'padding': '10px', 'borderRadius': '6px', 'fontWeight': 700,
                            'fontSize': '18px', 'marginBottom': '8px', 'opacity': 0.6 if weak else 1})]
    for i, name in enumerate(CLASSES):
        rows.append(html.Div(style={'position': 'relative', 'height': '26px', 'marginBottom': '4px',
                                    'background': '#f3f4f6', 'borderRadius': '4px', 'overflow': 'hidden'}, children=[
            html.Div(style={'width': f'{probs[i] * 100:.1f}%', 'height': '100%', 'background': COLORS[i]}),
            html.Div(f'{name}  {probs[i]:.3f}', style={'position': 'absolute', 'left': '8px', 'top': '3px',
                                                       'fontSize': '13px'})]))
    return rows, CLASSES[top]


def _rule_panel(found, top_class):
    if not found:
        return [html.Span('Rule check: no textbook pattern on the newest candles. ', style={'fontWeight': 600}),
                html.Span(f'Model says {top_class}; treat as low-signal.', style={'color': '#6b7280'})]
    agree = top_class in found
    return [html.Span(f"Rule check: {', '.join(found)}  ", style={'fontWeight': 600}),
            html.Span('agrees with model' if agree else f'differs from model ({top_class})',
                      style={'color': '#059669' if agree else '#b45309', 'fontWeight': 600})]


def _context_panel(visible, top_class):
    info = context.describe(visible, top_class)
    fit_color = {'fits': '#059669', 'against': '#b45309', 'neutral': '#6b7280'}[info['fit']]
    trend = 'n/a' if info['trend'] == 'n/a' else f"{info['trend']} ({info['trend_score']:+.1f} avg ranges)"
    volume = 'n/a' if info['volume'] == 'n/a' else f"{info['volume']} ({info['volume_ratio']:.1f}x avg)"
    return [html.Span(f'Trend before: {trend}  |  Volume: {volume}  |  ', style={'color': '#374151'}),
            html.Span({'fits': 'pattern fits the trend', 'against': 'pattern goes against the trend',
                       'neutral': 'no clear trend context'}[info['fit']], style={'color': fit_color, 'fontWeight': 600})]


def _table(visible):
    tail = visible.tail(MODEL_WINDOW)
    head = html.Tr([html.Th(c, style={'textAlign': 'right', 'padding': '4px 10px'}) for c in
                    ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']])
    body = [html.Tr([html.Td(market.format_time(r.Datetime, S['timeframe']), style={'padding': '4px 10px', 'textAlign': 'right'})] +
                    [html.Td(f'{v:,.2f}', style={'padding': '4px 10px', 'textAlign': 'right'}) for v in
                     (r.Open, r.High, r.Low, r.Close)] +
                    [html.Td(f'{int(r.Volume):,}', style={'padding': '4px 10px', 'textAlign': 'right'})])
            for r in tail.itertuples()]
    return html.Table([html.Thead(head), html.Tbody(body)], style={'borderCollapse': 'collapse', 'fontSize': '14px'})


def _history_table():
    rows = HISTORY.recent(10)
    if not rows:
        return html.Div('Nothing logged yet.', style={'color': '#6b7280', 'fontSize': '14px'})
    cols = [('candle_time', 'Candle'), ('ticker', 'Ticker'), ('mode', 'Mode'), ('timeframe', 'TF'), ('close', 'Close'),
            ('prediction', 'Model'), ('confidence', 'Conf.'), ('rule_patterns', 'Rules'), ('agrees', 'Agree')]
    head = html.Tr([html.Th(label, style={'textAlign': 'left', 'padding': '4px 10px'}) for _, label in cols])
    body = [html.Tr([html.Td(r.get(key) or '-', style={'padding': '4px 10px'}) for key, _ in cols]) for r in rows]
    return html.Table([html.Thead(head), html.Tbody(body)], style={'borderCollapse': 'collapse', 'fontSize': '13px'})


def _message(text, interval):
    empty = go.Figure().update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False)
    return empty, html.Span(text, style={'color': '#b91c1c', 'fontWeight': 600}), '', '', '', '', '', interval, no_update


@app.callback(
    Output('chart', 'figure'), Output('status', 'children'), Output('prediction', 'children'),
    Output('rule-check', 'children'), Output('context', 'children'), Output('model-view', 'src'),
    Output('table', 'children'), Output('tick', 'interval'), Output('history', 'children'),
    Input('tick', 'n_intervals'), Input('track-btn', 'n_clicks'), Input('ticker-input', 'n_submit'),
    Input('mode', 'value'), Input('timeframe', 'value'), Input('open-ticker', 'data'),
    State('ticker-input', 'value'), State('sid', 'data'), State('alert-conf', 'value'), State('alert-opts', 'value'),
)
def refresh(_n, _clicks, _submit, mode, timeframe, opened, ticker_text, sid, alert_conf, alert_opts):
    with _lock:
        S.use(sid)
        interval = _poll_ms(timeframe) if mode == 'live' else REPLAY_MS
        trigger = ctx.triggered_id
        if trigger == 'open-ticker':
            ticker_text = opened['ticker']
        try:
            # A page load (no trigger) or a mode/timeframe mismatch re-syncs the server with what the page shows
            if (trigger in (None, 'track-btn', 'ticker-input', 'mode', 'timeframe', 'open-ticker')
                    or S['mode'] != mode or S['timeframe'] != timeframe
                    or (S['df'] is None and not S['error'])):
                S['error'] = ''
                _start(ticker_text, mode, timeframe)
            elif S['df'] is None:  # last attempt failed; wait for the user instead of hammering Yahoo
                return _message(S['error'], interval)
            elif mode == 'replay':
                S['cursor'] = min(S['cursor'] + 1, len(S['df']))
            elif time.monotonic() - S['fetched'] > interval / 1000 - 1:
                try:
                    S.update(df=market.fetch_candles(S['ticker'], S['timeframe']), fetched=time.monotonic(), warning='')
                except market.MarketDataError as exc:  # keep showing the last good data
                    S['warning'] = f'refresh failed: {exc}'
            visible = _visible()
            if len(visible) < render.N_CANDLES:
                return _message(f'Only {len(visible)} closed candles so far; need {render.N_CANDLES}.', interval)
            probs, src = _predict(visible)
        except market.MarketDataError as exc:
            S.update(df=None, error=str(exc))
            return _message(str(exc), interval)
        except FileNotFoundError as exc:
            return _message(str(exc), interval)

        panel, top_class = _prediction_panel(probs)
        found = rules.detect_frame(visible)
        HISTORY.log(S['ticker'], S['mode'], visible['Datetime'].iloc[-1], visible['Close'].iloc[-1], top_class,
                    max(probs), found, timeframe=S['timeframe'])
        if S['mode'] == 'live':  # replayed candles are not real signals
            _notify([scanner.build_row(S['ticker'], S['timeframe'], visible, probs, S['df'])], alert_conf, alert_opts)
        return (_figure(visible), _status(visible), panel, _rule_panel(found, top_class),
                _context_panel(visible, top_class), src, _table(visible), interval, _history_table())


@app.callback(Output('export-download', 'data'), Input('export-btn', 'n_clicks'), prevent_initial_call=True)
def export_history(_clicks):
    return dcc.send_file(HISTORY.path) if HISTORY.exists() else no_update


def _scan_table(rows):
    cols = ['Ticker', 'Model', 'Conf.', 'Rules', 'Agree', 'Trend', 'Volume', 'Context', 'Market', 'Last candle', 'Close', '']
    head = html.Tr([html.Th(c, style={'textAlign': 'left', 'padding': '4px 10px'}) for c in cols])
    body = []
    for r in rows:
        button = html.Button('Open', id={'type': 'open-ticker', 'ticker': r['ticker']}, n_clicks=0,
                             style={'cursor': 'pointer'})
        cell = {'padding': '4px 10px'}
        if r['error']:
            body.append(html.Tr([html.Td(r['ticker'], style={**cell, 'fontWeight': 600}),
                                 html.Td(r['error'], colSpan=10, style={**cell, 'color': '#b91c1c'}),
                                 html.Td(button, style=cell)]))
            continue
        tint = '#ecfdf5' if r['agrees'] else 'transparent'
        body.append(html.Tr(style={'background': tint}, children=[
            html.Td(r['ticker'], style={**cell, 'fontWeight': 600}),
            html.Td(r['prediction'], style={**cell, 'background': COLORS[CLASSES.index(r['prediction'])]}),
            html.Td(f"{r['confidence']:.1%}", style=cell),
            html.Td(', '.join(r['rules']) or '-', style=cell),
            html.Td('yes' if r['agrees'] else 'no', style=cell),
            html.Td(r['trend'], style=cell),
            html.Td(r['volume'] if r['volume'] == 'n/a' else f"{r['volume']} ({r['volume_ratio']:.1f}x)", style=cell),
            html.Td(r['fit'], style={**cell, 'color': {'fits': '#059669', 'against': '#b45309'}.get(r['fit'], '#6b7280')}),
            html.Td(r['market'], style=cell),
            html.Td(market.format_time(r['candle_time'], r['timeframe']), style=cell),
            html.Td(f"{r['close']:,.2f}", style=cell),
            html.Td(button, style=cell),
        ]))
    return html.Table([html.Thead(head), html.Tbody(body)], style={'borderCollapse': 'collapse', 'fontSize': '14px'})


@app.callback(Output('scan-results', 'children'), Output('scan-status', 'children'),
              Input('scan-btn', 'n_clicks'), Input('scan-tick', 'n_intervals'), State('scan-input', 'value'),
              State('timeframe', 'value'), State('scan-fit', 'value'), State('alert-conf', 'value'),
              State('alert-opts', 'value'), prevent_initial_call=True)
def run_scan(_clicks, _n, text, timeframe, fit_only, alert_conf, alert_opts):
    tickers, rejected, truncated = scanner.parse_tickers(text)
    notes = []
    if rejected:
        notes.append(f"ignored: {', '.join(rejected)}")
    if truncated:
        notes.append(f'only the first {scanner.MAX_TICKERS} are scanned')
    if not tickers:
        return no_update, 'Enter at least one valid ticker.'
    started = time.monotonic()
    rows = scanner.scan(tickers, _infer, timeframe=timeframe)
    with _lock:
        for r in rows:
            if not r['error']:
                HISTORY.log(r['ticker'], 'scan', r['candle_time'], r['close'], r['prediction'], r['confidence'],
                            r['rules'], timeframe=timeframe)
    _notify(rows, alert_conf, alert_opts)
    ok = sum(1 for r in rows if not r['error'])
    status = (f'{ok}/{len(rows)} scanned ({timeframe}) in {time.monotonic() - started:.1f} s '
              f'at {time.strftime("%H:%M:%S")}')
    if fit_only:
        shown = [r for r in rows if r['error'] or r['fit'] == 'fits']
        notes.append(f"{len(shown) - sum(1 for r in shown if r['error'])} fit the trend")
        rows = shown
    return _scan_table(rows), '  |  '.join([status] + notes)


@app.callback(Output('scan-tick', 'disabled'), Input('scan-auto', 'value'))
def toggle_auto_scan(value):
    return 'auto' not in (value or [])


@app.callback(Output('open-ticker', 'data'), Input({'type': 'open-ticker', 'ticker': ALL}, 'n_clicks'),
              prevent_initial_call=True)
def open_ticker(_clicks):
    trigger = ctx.triggered[0] if ctx.triggered else None
    if not trigger or not trigger['value']:  # re-rendered buttons start at 0 and must not count as clicks
        return no_update
    return {'ticker': ctx.triggered_id['ticker'], 'clicks': trigger['value']}


@app.callback(Output('ticker-input', 'value'), Input('open-ticker', 'data'), prevent_initial_call=True)
def show_opened_ticker(opened):
    return opened['ticker'] if opened else no_update


@app.callback(Output('alerts-feed', 'children'), Output('alert-signal', 'data'), Output('telegram-status', 'children'),
              Input('alert-tick', 'n_intervals'))
def render_alerts(_n):
    items = FEED.recent(10)
    feed = ([html.Div(f"{a['time']}  {a['message']}", style={'fontSize': '14px', 'padding': '2px 0'}) for a in items]
            or html.Div('No alerts yet.', style={'fontSize': '14px', 'color': '#6b7280'}))
    if alerts.telegram_credentials():
        telegram = 'Telegram: configured.' + (f" Last send failed: {_telegram['error']}" if _telegram['error'] else '')
    else:
        telegram = 'Telegram: not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID before starting the app.'
    return feed, {'total': FEED.total, 'latest': items[0]['message'] if items else ''}, telegram


app.clientside_callback(
    """function(signal, opts) {
        if (!signal) { return window.dash_clientside.no_update; }
        if (window.__alertSeen === undefined) { window.__alertSeen = signal.total; }  // skip alerts from before page load
        if (signal.total > window.__alertSeen) {
            window.__alertSeen = signal.total;
            if (opts && opts.includes('browser') && window.Notification && Notification.permission === 'granted') {
                new Notification('Candlestick alert', {body: signal.latest});
            }
        }
        return window.dash_clientside.no_update;
    }""",
    Output('alert-sink-1', 'data'), Input('alert-signal', 'data'), State('alert-opts', 'value'))

app.clientside_callback(
    """function(opts) {
        if (opts && opts.includes('browser') && window.Notification && Notification.permission === 'default') {
            Notification.requestPermission();
        }
        return window.dash_clientside.no_update;
    }""",
    Output('alert-sink-2', 'data'), Input('alert-opts', 'value'), prevent_initial_call=True)


if __name__ == '__main__':
    predictor.load_model()
    print('Open http://127.0.0.1:8050  (Ctrl+C to stop)')
    app.run(host='127.0.0.1', port=8050, debug=False)
