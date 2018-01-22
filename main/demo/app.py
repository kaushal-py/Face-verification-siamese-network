from flask import Flask, render_template
from eyc_dataset_loader import EycDataset
from torch.utils.data import DataLoader,Dataset

app = Flask(__name__)

dataset = EycDataset()

train_dataloader = DataLoader(dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=1)


@app.route('/')
def homepage():
    
    return render_template("main.html", images=train_dataloader[:5])

if __name__ == "__main__":
    app.run()