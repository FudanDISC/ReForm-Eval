import argparse
from models import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, help='the model name to load')
    args = parser.parse_args()

    model = get_model(args.model_name)
    print(model)
    return model

if __name__=='__main__':
    main()
