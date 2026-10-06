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

import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from PIL import Image

import market
import predictor
import render
import rules

CLASSES = predictor.CLASSES
COLORS = ['#ebdc9c', '#8ccfa6', '#c1abec', '#e8a3ca', '#ff8080']
LIVE_POLL_MS = 30_000
REPLAY_MS = 1_000
MODEL_WINDOW = 8  # candles the ViT crop covers
START_TICKER = 'AMZN'

_lock = threading.Lock()
S = {'ticker': None, 'mode': 'live', 'df': None, 'cursor': 0, 'fetched': 0.0, 'warning': '', 'error': '', 'cache': (None, None)}

app = Dash(__name__, title='Candlestick Pattern Tracker')

PAGE = {'fontFamily': 'system-ui, Segoe UI, sans-serif', 'maxWidth': '1280px', 'margin': '0 auto', 'padding': '16px',
        'color': '#1f2937'}
CARD = {'background': '#fff', 'border': '1px solid #e5e7eb', 'borderRadius': '10px', 'padding': '14px'}

app.layout = html.Div(style=PAGE, children=[
    html.H2('Candlestick Pattern Tracker', style={'margin': '0 0 12px'}),
    html.Div(style={'display': 'flex', 'gap': '10px', 'alignItems': 'center', 'flexWrap': 'wrap'}, children=[
        dcc.Input(id='ticker-input', type='text', value=START_TICKER, placeholder='Ticker, e.g. AAPL',
                  debounce=False, n_submit=0, style={'padding': '8px 10px', 'width': '180px', 'fontSize': '16px'}),
        html.Button('Track', id='track-btn', n_clicks=0,
                    style={'padding': '8px 18px', 'fontSize': '16px', 'cursor': 'pointer'}),
        dcc.RadioItems(id='mode', value='live', inline=True, inputStyle={'marginRight': '4px', 'marginLeft': '12px'},
                       options=[{'label': 'Live (polls Yahoo every 30 s)', 'value': 'live'},
                                {'label': 'Replay last session (1 candle/s)', 'value': 'replay'}]),
    ]),
    html.Div(id='status', style={'margin': '10px 0', 'fontSize': '14px'}),
    html.Div(style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}, children=[
        html.Div(style={**CARD, 'flex': '3 1 640px', 'minWidth': '0'},
                 children=dcc.Graph(id='chart', config={'displayModeBar': False}, style={'height': '440px'})),
        html.Div(style={**CARD, 'flex': '1 1 320px'}, children=[
            html.Div('ViT prediction', style={'fontWeight': 600, 'marginBottom': '8px'}),
            html.Div(id='prediction'),
            html.Div(id='rule-check', style={'marginTop': '12px', 'fontSize': '14px'}),
            html.Div('What the model sees', style={'fontWeight': 600, 'margin': '14px 0 6px'}),
            html.Img(id='model-view', style={'imageRendering': 'pixelated', 'border': '1px solid #e5e7eb'}),
        ]),
    ]),
    html.Div(style={**CARD, 'marginTop': '14px', 'overflowX': 'auto'}, children=[
        html.Div(f'Last {MODEL_WINDOW} candles (the ones the model reads) - compare with Yahoo or your broker',
                 style={'fontWeight': 600, 'marginBottom': '8px'}),
        html.Div(id='table'),
    ]),
    html.Div('Educational demo, not financial advice. The model has no "no pattern" class: it always picks one of '
             'five, so read the confidence and the rule check.',
             style={'marginTop': '12px', 'fontSize': '12px', 'color': '#6b7280'}),
    dcc.Interval(id='tick', interval=REPLAY_MS, n_intervals=0),
])


def _start(ticker_text, mode):
    """(Re)load the ticker. Raises MarketDataError with a user-readable message."""
    ticker = market.normalize_ticker(ticker_text)
    df = market.fetch_candles(ticker)
    S.update(ticker=ticker, mode=mode, df=df, cursor=market.MIN_CANDLES, fetched=time.monotonic(), warning='',
             cache=(None, None))


def _visible():
    df = S['df']
    return df.iloc[:S['cursor']] if S['mode'] == 'replay' else market.closed_only(df)


def _status(visible):
    last = visible['Datetime'].iloc[-1]
    state, age = market.market_status(S['df'])
    if S['mode'] == 'replay':
        text = f"REPLAY {S['ticker']} - candle {len(visible)}/{len(S['df'])} - {last:%Y-%m-%d %H:%M %Z}"
        color = '#2563eb'
    elif state == 'live':
        text = f"LIVE {S['ticker']} - last closed candle {last:%H:%M %Z} ({age:.0f} min ago; Yahoo may be delayed)"
        color = '#059669'
    else:
        text = (f"MARKET CLOSED {S['ticker']} - showing last session ending {last:%Y-%m-%d %H:%M %Z}. "
                f'Switch to Replay to watch it play out.')
        color = '#b45309'
    children = [html.Span(text, style={'color': color, 'fontWeight': 600})]
    if S['warning']:
        children.append(html.Span(f"  |  {S['warning']}", style={'color': '#b91c1c'}))
    return children


def _figure(visible):
    data = visible.iloc[-render.N_CANDLES:]
    fig = go.Figure(go.Candlestick(x=data['Datetime'], open=data['Open'], high=data['High'], low=data['Low'],
                                   close=data['Close'], increasing_line_color='#3D9970', decreasing_line_color='#FF4136'))
    half = (data['Datetime'].iloc[1] - data['Datetime'].iloc[0]) / 2
    fig.add_vrect(x0=data['Datetime'].iloc[-MODEL_WINDOW] - half, x1=data['Datetime'].iloc[-1] + half,
                  fillcolor='#6366f1', opacity=0.10, line_width=0,
                  annotation_text='model window', annotation_position='top left', annotation_font_size=11)
    fig.update_layout(margin=dict(l=50, r=20, t=20, b=40), xaxis_rangeslider_visible=False, uirevision=S['ticker'],
                      height=420, template='plotly_white', showlegend=False)
    return fig


def _predict(visible):
    key = (S['ticker'], visible['Datetime'].iloc[-1])
    if S['cache'][0] == key:
        return S['cache'][1]
    data = visible.iloc[-render.N_CANDLES:]
    image = render.render_chart(data['Open'], data['High'], data['Low'], data['Close'], list(data['Datetime']))
    probs = predictor.predict(image)
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


def _rule_panel(visible, top_class):
    candles = [tuple(r) for r in visible[['Open', 'High', 'Low', 'Close']].tail(3).itertuples(index=False)]
    found = rules.detect(candles)
    if not found:
        return [html.Span('Rule check: no textbook pattern on the newest candles. ', style={'fontWeight': 600}),
                html.Span(f'Model says {top_class}; treat as low-signal.', style={'color': '#6b7280'})]
    agree = top_class in found
    return [html.Span(f"Rule check: {', '.join(found)}  ", style={'fontWeight': 600}),
            html.Span('agrees with model' if agree else f'differs from model ({top_class})',
                      style={'color': '#059669' if agree else '#b45309', 'fontWeight': 600})]


def _table(visible):
    tail = visible.tail(MODEL_WINDOW)
    head = html.Tr([html.Th(c, style={'textAlign': 'right', 'padding': '4px 10px'}) for c in
                    ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']])
    body = [html.Tr([html.Td(f'{r.Datetime:%H:%M}', style={'padding': '4px 10px', 'textAlign': 'right'})] +
                    [html.Td(f'{v:,.2f}', style={'padding': '4px 10px', 'textAlign': 'right'}) for v in
                     (r.Open, r.High, r.Low, r.Close)] +
                    [html.Td(f'{int(r.Volume):,}', style={'padding': '4px 10px', 'textAlign': 'right'})])
            for r in tail.itertuples()]
    return html.Table([html.Thead(head), html.Tbody(body)], style={'borderCollapse': 'collapse', 'fontSize': '14px'})


def _message(text, interval):
    empty = go.Figure().update_layout(template='plotly_white', xaxis_visible=False, yaxis_visible=False)
    return empty, html.Span(text, style={'color': '#b91c1c', 'fontWeight': 600}), '', '', '', '', interval


@app.callback(
    Output('chart', 'figure'), Output('status', 'children'), Output('prediction', 'children'),
    Output('rule-check', 'children'), Output('model-view', 'src'), Output('table', 'children'),
    Output('tick', 'interval'),
    Input('tick', 'n_intervals'), Input('track-btn', 'n_clicks'), Input('ticker-input', 'n_submit'),
    Input('mode', 'value'), State('ticker-input', 'value'),
)
def refresh(_n, _clicks, _submit, mode, ticker_text):
    with _lock:
        interval = LIVE_POLL_MS if mode == 'live' else REPLAY_MS
        trigger = ctx.triggered_id
        try:
            if trigger in ('track-btn', 'ticker-input', 'mode') or (S['df'] is None and not S['error']):
                S['error'] = ''
                _start(ticker_text, mode)
            elif S['df'] is None:  # last attempt failed; wait for the user instead of hammering Yahoo
                return _message(S['error'], interval)
            elif mode == 'replay':
                S['cursor'] = min(S['cursor'] + 1, len(S['df']))
            elif time.monotonic() - S['fetched'] > LIVE_POLL_MS / 1000 - 1:
                try:
                    S.update(df=market.fetch_candles(S['ticker']), fetched=time.monotonic(), warning='')
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
        return (_figure(visible), _status(visible), panel, _rule_panel(visible, top_class), src, _table(visible),
                interval)


if __name__ == '__main__':
    predictor.load_model()
    print('Open http://127.0.0.1:8050  (Ctrl+C to stop)')
    app.run(host='127.0.0.1', port=8050, debug=False)
