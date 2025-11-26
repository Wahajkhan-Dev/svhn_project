from dash import Dash, dcc, html, Input, Output
import os, base64

OUT = 'outputs'
app = Dash(__name__)

def encode_img(path):
    try:
        with open(path,'rb') as f:
            return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
    except Exception:
        return None

def list_imgs(folder, prefix='img_'):
    if not os.path.isdir(folder):
        return []
    return sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.startswith(prefix)])[:30]

app.layout = html.Div([
    html.H3('SVHN — Results Dashboard (short)'),
    html.Div([
        html.Button('Show misclassified', id='mis-btn'),
        html.Button('Show Grad-CAMs', id='cam-btn'),
    ], style={'marginBottom':'10px'}),
    html.Div(id='grid', style={'display':'flex','flexWrap':'wrap'})
])

@app.callback(Output('grid','children'), [Input('mis-btn','n_clicks'), Input('cam-btn','n_clicks')])
def update(mis, cam):
    ctx = None
    try:
        from dash import callback_context
        ctx = callback_context
    except Exception:
        pass
    if not ctx or not ctx.triggered:
        return ''
    name = ctx.triggered[0]['prop_id'].split('.')[0]
    if name == 'mis-btn':
        folder = os.path.join(OUT, 'mis_examples')
        imgs = list_imgs(folder, prefix='mis_')
    else:
        folder = os.path.join(OUT, 'gradcam')
        imgs = list_imgs(folder, prefix='img_')
    if not imgs:
        return html.Div(f'No images found in {folder}. Run evaluate.py and gradcam.py and refresh.', style={'color':'red','padding':'10px'})
    thumbs = []
    for p in imgs:
        src = encode_img(p)
        if src:
            thumbs.append(html.Img(src=src, style={'height':'120px','margin':'4px'}))
    if not thumbs:
        return html.Div('Images exist but failed to load (check file permissions).', style={'color':'red'})
    return thumbs

if __name__ == '__main__':
    app.run(debug=True)
