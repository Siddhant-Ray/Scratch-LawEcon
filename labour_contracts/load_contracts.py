import os, json 
import pandas as pd 
from tqdm import tqdm

def get_text(cur_art):
    output = ""
    for line in cur_art["lines"]:
        output = output + " " + (line["text"])
    return output


def load_articles(path):
    files = os.listdir(path)
    for fn in tqdm(files):
        with open(os.path.join(path, fn)) as f:
            contract_data = json.loads(f.read())
            contract_id = contract_data["contract_id"]
            art_list = contract_data["articles"]
            for art_num, cur_art in enumerate(art_list):
                try:
                    header = cur_art["header"]["text"].lower()
                    text = get_text(cur_art)
                    yield (contract_id,header,text)
                except:
                    pass

def main():
    path = "/cluster/work/lawecon/Work/dominik/powerparser/output_canadian_new/01_artsplit"
    load_generator = load_articles(path)
    for value in load_generator:
        print(value)
        break

    df = pd.DataFrame(load_generator)
    print(df.head())

    save_path = "labour_contracts/data/relatio_formatted.csv"
    df.to_csv(path, index=False)

if __name__=="__main__":
    main()