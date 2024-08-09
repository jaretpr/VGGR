import argparse
from augment import augment
from train import train_model
from predict import predict
from predict import demo
from train import device_options
from train import model_options
from train import cnn_v1
from train import cnn_v2
from train import cnn_v3
from colors import color


if __name__ == '__main__':
    # Commands
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--predict', action='store_true', help='Predict mode.')
    parser.add_argument('--train', action='store_true', help='Train mode.')
    parser.add_argument('--demo', action='store_true', help='Test-Set Demo.')
    parser.add_argument('--augment', action='store_true', help='Augmentation of the training dataset.')
    parser.add_argument('-i', '--input', type=str, required=False, help='Image file input for prediction / inference.')
    parser.add_argument('-d', '--device', type=str, choices=device_options, default='cpu', required=False, help='Device selection for training and inference / predicting.')
    parser.add_argument('-m', '--model', type=str, choices=model_options, required=False, default='cnn_v1', help='Model selection for --train, --predict, --demo.')
    args = parser.parse_args() 
    
    
    # Data Augmentation
    if args.augment == True:
        augment()
    
    # Train mode
    elif args.train == True and args.model != None:
        if args.model == 'cnn_v1':
            train_model(model=cnn_v1, batch_size=400, learn_rate=0.001, device=args.device)
        elif args.model == 'cnn_v2':
            train_model(model=cnn_v2, batch_size=160, learn_rate=0.01, device=args.device)
        elif args.model == 'cnn_v3':
            train_model(model=cnn_v3, batch_size=400, learn_rate=0.001, device=args.device)
        else:
            print(f'{color.RED}Invalid model input!{color.END}')
    
    # Predict mode
    elif args.predict == True and args.model != None:
        if args.input != None:
            if args.model == 'cnn_v1':
                predict(model=cnn_v1, device=args.device, input_file = args.input)
            elif args.model == 'cnn_v2':
                predict(model=cnn_v2, device=args.device, input_file = args.input)
            elif args.model == 'cnn_v3':
                predict(model=cnn_v3, device=args.device, input_file = args.input)
        else:
            print(f'{color.RED}Invalid input file!{color.END}')
    
    # Demo / Test set mode
    elif args.demo == True and args.model != None:
        if args.model == 'cnn_v1':
            demo(model=cnn_v1, device=args.device)
        elif args.model == 'cnn_v2':
            demo(model=cnn_v2, device=args.device)
        elif args.model == 'cnn_v3':
            demo(model=cnn_v3, device=args.device)

