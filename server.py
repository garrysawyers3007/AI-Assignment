from flask import Flask, request, jsonify, abort, Response, render_template, session, url_for
from werkzeug.utils import redirect
from model import GKT, MultiHeadAttention, ScaledDotProductAttention, MLP, EraseAddGate
import torch
from utils import *
import pandas as pd
app = Flask(__name__)

model = None 

@app.route('/', methods=['POST', 'GET'])
def get_probs():
    if (request.method == 'GET'):
        return render_template('index.html')

    if(request.method == 'POST'):
        user_id = int(request.form['user_id'])
        skill_id = int(request.form['skill_id'])
        avg_prob = get_prob(user_id, skill_id)
        session['avg_prob'] = avg_prob
        return redirect(url_for('get_result'))
    # return jsonify({
    #     'probability': avg_prob
    # })

def get_prob(user_id, skill_id):
    df = pd.read_csv('assistment_test15.csv')
    data_loader, id_map = load_data(df[(df['user_id']==user_id)])
    for _, (features, questions) in enumerate(data_loader):
        pred_res, _, _, _ = model(features, questions)
    try:
        temp = torch.where(questions==id_map[skill_id])[1]
    except KeyError as error: 
        abort(Response('Invalid Skill ID - {}'.format(error), status= 400))
    x = [pred_res[0][t-1].item() for t in temp]
    avg_prob = sum(x)/len(x)
    return avg_prob

@app.route('/result', methods=['GET'])
def get_result():
    if (request.method == 'GET'):
        print(session['avg_prob'])
        return render_template('result.html', context={'avg_prob': session['avg_prob']})

if __name__ == '__main__':
    app.secret_key = '8f42a73054b1749f8f58848be5e6502c'
    model = torch.load('model_50.pt')
    app.run(debug=True)