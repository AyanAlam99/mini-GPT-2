import os , torch 
from config import save_dir


def save_model(model, counter):
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f'model_iter_{counter}.pt')
    torch.save(model.state_dict(), model_path)
    print(f" Model saved at iteration {counter}")

def load_model(model, path):
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint)
    return model
