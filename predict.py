import torch
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.data import DataLoader
from data.datasets import test_data
from data.datasets import CustomTestDataset
from data.datasets import train_data
from data.datasets import data_resize
from data.datasets import data_transform
from colors import color
import os
import base64


test_dataloader = DataLoader(test_data, batch_size=1000, shuffle=False)


def predict(model, device, input_file):
    with torch.inference_mode():    
        if os.path.isfile(f'./models/saved/{model.__class__.__name__}.pth'):   
            # Loading trained model
            checkpoint = torch.load(f'./models/saved/{model.__class__.__name__}.pth')
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            model.eval()
            
            if input_file != None:
                custom_test_data = CustomTestDataset(data_resize, data_transform, input_file)
                custom_test_dataloader = DataLoader(custom_test_data, batch_size=1)
            
                for img in custom_test_dataloader:
                    output = model(img.to(device))
                    print(f'\nGenre Prediction: {color.BLUE}{train_data.genre[output.argmax(dim=1).item()]}{color.END}')
            else:
                print(f'\n{color.RED}No input file specified!{color.END}')
        else:
            print(f'\n{color.RED}Trained model does not exist!{color.END}')


def demo(model, device):
    font = 'style="font-family:calibri"'
    html_file = open(f'./results/{model.__class__.__name__}_results.html', 'w')
    with torch.inference_mode():    
        if os.path.isfile(f'./models/saved/{model.__class__.__name__}.pth'):   
            # loading trained model
            checkpoint = torch.load(f'./models/saved/{model.__class__.__name__}.pth')
            best_epoch = checkpoint['best_epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model = model.to(device)
            model.eval()
            file_index = 1
            correct = 0
            print(f'\nPredicting with {model.__class__.__name__}...\n')
            for img, label, game, files in test_dataloader:
                output = model(img.to(device))
                
                for sample_pred, sample_label, sample_game, sample_file in zip(output, label, game, files):
                    actual_label = train_data.genre[sample_label.item()]
                    prediction  = train_data.genre[sample_pred.argmax().item()]
                    data_uri = base64.b64encode(open(sample_file, 'rb').read()).decode('utf-8')
                    img_tag = '<img src="data:img/png;base64,{0}" width="752" height="423">'.format(data_uri)
                    
                    if prediction == actual_label:
                        correct += 1
                        clr = color.BLUE
                        clr_html = 'color:blue;'
                    else:
                        clr = color.RED
                        clr_html = 'color:red;'
                    
                    prediction_shell = f'{clr}{prediction}{color.END}'
                    prediction_html = f'<strong style="{clr_html};" {font}>{prediction}</strong>'
                    actual_label_shell = f'{clr}{actual_label}{color.END}\n'
                    actual_label_html = f'<strong style="{clr_html};" {font}>{actual_label}</strong>'
                    
                    print(f'{file_index}.\nGame: {sample_game}')
                    print(f'Prediction: {prediction_shell}')
                    print(f'Actual: {actual_label_shell}')
                    html_file.write(f'<br /><h3 {font}>&emsp;{file_index}.</h3> <h3 {font}>&emsp;Game: {sample_game}&emsp;|&emsp;Prediction: {prediction_html}&emsp;|&emsp;Actual: {actual_label_html}</h3> {img_tag} <br />')
                    
                    file_index += 1
            
            pred_list = [x.item() for x in output.argmax(dim=1)]
            actual_list = [y.item() for y in label]
        
        else:
            print(f'\n{color.RED}Trained model does not exist!{color.END}')
            exit()
            
        f1 = multiclass_f1_score(torch.tensor(pred_list).to(device), torch.tensor(actual_list).to(device))
        html_file.write(f'<br /><h3 {font}>&emsp;Correct predictions: {correct} / {len(test_data)}</h3>')
        html_file.write(f'<h3 {font}>&emsp;F1 score: {f1}</h3>')
        html_file.write(f'<h3 {font}>&emsp;Best epoch: {best_epoch}</h3>')
        html_file.close()
        print(f'Correct predictions: {correct} / {len(test_data)}')
        print(f'F1 score: {f1}')

