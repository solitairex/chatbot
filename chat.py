import random
import json
import torch

import webview
from model import NeuralNet
from ntlk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE, map_location=device, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

class Api:
    def get_bot_response(self, sentence):
        # Check if the user typed "quit"
        if sentence.strip().lower() == "quit":
            return "Bot shutting down. Goodbye!"
            exit()  # Exit the application

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    response = random.choice(intent['responses'])
                    # Adding links to specific game titles
                    response = response.replace('Hades', '<a href="https://www.amazon.com/dp/B08L8LG4M3">Hades</a>')
                    response = response.replace('Ghost of Tsushima', '<a href="https://www.amazon.com/s?k=ghost+of+tsushima&crid=2VV0LGHDGS32&sprefix=ghost+of+tsushima%2Caps%2C300&ref=nb_sb_noss_1">Ghost of Tsushima</a>')
                    response = response.replace('Nike', '<a href="https://www.amazon.com/s?k=Nike">Nike</a>')
                    response = response.replace('Adidas', '<a href="https://www.amazon.com/s?k=Adidas">Adidas</a>')
                    response = response.replace('H&M', '<a href="https://www.amazon.com/s?k=vêtmentsH&M">H&M</a>')
                    response = response.replace('Mario Kart 8 Deluxe', '<a href="https://www.amazon.com/dp/B01N1037CV">Mario Kart 8 Deluxe</a>')
                    response = response.replace('Seigneur des Anneaux', '<a href="https://www.amazon.com/s?k=seigneurdesanneaux">Seigneur des Anneaux</a>')
                    response = response.replace('1984', '<a href="https://www.amazon.com/s?k=1984">1984</a>')
                    response = response.replace('Harry Potter', '<a href="https://www.amazon.com/s?k=harrypotter">Harry Potter</a>')
                    response = response.replace('Le Petit Prince', '<a href="https://www.amazon.com/s?k=petitprince">Le Petit Prince</a>')
                    response = response.replace('Game of Thrones', '<a href="https://www.amazon.com/s?k=gamethrones">Game of Thrones</a>')
                    response = response.replace('Great Gatsby', '<a href="https://www.amazon.com/s?k=greatgatsby">Great Gatsby</a>')
                    response = response.replace('Dell XPS 13', '<a href="https://www.amazon.com/s?k=xps13">Dell XPS 13</a>')
                    response = response.replace('MacBook Pro', '<a href="https://www.amazon.com/s?k=macbook">MacBook Pro</a>')
                    response = response.replace('HP Omen', '<a href="https://www.amazon.com/s?k=HPomen">HP Omen</a>')

            return response
        return "J'ai pas bien compris, repetez s'il vous plait!"


def open_html_window():
    api = Api()
    window = webview.create_window('Sam', 'chat.html', js_api=api, width=1200, height=800)
    webview.start()

if __name__ == "__main__":
    open_html_window()