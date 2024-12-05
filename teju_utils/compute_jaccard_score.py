from collections import defaultdict
import json

def jaccard_index(list1, list2):
    set1, set2 = set(list1), set(list2)
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def average_jaccard_index(lists):
    num_lists = len(lists)
    total_jaccard = 0
    count = 0
    for i in range(num_lists):
        for j in range(i + 1, num_lists):
            total_jaccard += jaccard_index(lists[i], lists[j])
            count += 1
    return total_jaccard / count if count > 0 else 0

def merge_json_objects(json_list):
    combined_dict = defaultdict(list)
    
    for json_obj in json_list:
        for key, value in json_obj.items():
            combined_dict[key].append(value)
    
    return dict(combined_dict)

if __name__ == "__main__":
    import glob
    
    data="PBMC_CTL"
    
    for data in ["PBMC", "PBMC_CTL", "BoneMarrow"]:
        files = glob.glob(f'/p/home/jusers/afonja1/juwels/teju/GRouNdGAN/data/processed/{data}/causal_graph_**.json', recursive=True)
        save_json=f'exp/grnboost2/{data}_jaccard.json'
        
        jsons = []
        for file in files:
            with open(file, 'r') as f:
                json_file = json.load(f)
            jsons.append(json_file)
        
        # print(jsons)
        
        jsons_merged_dict = merge_json_objects(jsons)
        # print(jsons_merged_dict)
        # print(jsons_merged_dict.keys())
        
        all_keys_jaccard = {}
        for key, values in jsons_merged_dict.items():
            score=average_jaccard_index(values)
            all_keys_jaccard[key] = score

        print(len(all_keys_jaccard))
        avg_over_genes = sum(all_keys_jaccard.values()) / len(all_keys_jaccard.keys())
        
        all_keys_jaccard["*overall_average"] = avg_over_genes
        
        with open(save_json, 'w') as f:
            json.dump(all_keys_jaccard, f, sort_keys=True, indent=2)