from flask import Flask, request, jsonify, abort, Response
from model import GKT, MultiHeadAttention, ScaledDotProductAttention, MLP, EraseAddGate
import torch
from utils import *
import pandas as pd
app = Flask(__name__)

model = None 

@app.route('/', methods=['POST'])
def get_probs():
    if(request.method=='POST'):
        json_body= request.get_json()
        user_id = json_body.get('user_id')
        skill_id = json_body.get('skill_id')
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
    return jsonify({
        'probability': avg_prob
    })

if __name__ == '__main__':
    model = torch.load('model_50.pt')
    app.run(debug=True)