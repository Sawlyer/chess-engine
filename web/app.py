from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import chess
import os
import torch
import pickle
import sys
import random
import json
import torch.nn as nn
import torch.nn.functional as F

# Ajouter le dossier engines au path pour pouvoir importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from engines.model import ChessModel
from engines.auxiliary_func import board_to_matrix

app = Flask(__name__, template_folder='templates')
CORS(app)

# Charger le modèle et le mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

with open("../models/heavy_move_to_int", "rb") as file:
    move_to_int = pickle.load(file)

# Créer le modèle avec 13 plans d'entrée et 1394 classes comme dans le modèle sauvegardé
model = ChessModel(num_classes=len(move_to_int), n_input_planes=13).to(device)
model.load_state_dict(torch.load("../models/noob.pth"))
model.eval()

int_to_move = {v: k for k, v in move_to_int.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'js'), filename)

@app.route('/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'css'), filename)

@app.route('/img/<path:filename>')
def serve_img(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), 'img'), filename)

@app.route('/models/<path:filename>')
def serve_model(filename):
    return send_from_directory(os.path.join(os.path.dirname(__file__), '..', 'models'), filename)

@app.route('/move', methods=['POST'])
def move():
    data = request.json
    board = chess.Board(data['fen'])

    # Préparer l'entrée
    matrix   = board_to_matrix(board)
    X_tensor = torch.tensor(matrix, dtype=torch.float32).unsqueeze(0).to(device)

    # Inférence
    with torch.no_grad():
        logits, _ = model(X_tensor)                   # ← ici
        move_idx  = logits.argmax(dim=1).item()       # ← et ici
        ai_move   = int_to_move[move_idx]

    # Vérifier si le coup est légal
    try:
        board.push_uci(ai_move)
    except:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return jsonify({'move': None, 'fen': board.fen(), 'status': 'gameover'})
        ai_move = random.choice(legal_moves).uci()
        board.push_uci(ai_move)

    return jsonify({'move': ai_move, 'fen': board.fen(), 'status': 'ok'})


if __name__ == '__main__':
    app.run(debug=True, port=5000) 