import pandas as pd
from itertools import combinations

# dataset path
path = "dataset/apriori_data.csv"

class apriori():
    def __init__(self, file, min_support, min_confidence):
        file = file.applymap(lambda x: str(x).strip().upper() == 'TRUE')
        
        # 將每一筆交易中的 'True' 商品提取出來作為一個交易
        self.transactions = file.apply(lambda row: set(file.columns[row]), axis=1)
        self.min_support = min_support
        self.min_confidence = min_confidence
    
    # calculate support for individual items (1-itemsets)
    def calculate_support(self, itemset):
        count = 0
        for transaction in self.transactions:
            if set(itemset).issubset(set(transaction)):
                count += 1
        return count / len(self.transactions)

    # function to get all unique items from transactions
    def get_unique_items(self):
        items = set()
        for transaction in self.transactions:
            items.update(transaction)
        return sorted(items)
    
    def generate_association_rules(self, frequent_itemsets):
        rules = []
        # For each frequent itemset, try to split it into two parts (X -> Y)
        for itemset in frequent_itemsets:
            if len(itemset) > 1:
                for i in range(1, len(itemset)):
                    for antecedent in combinations(itemset, i):  # X part
                        antecedent = set(antecedent)
                        consequent = itemset - antecedent  # Y part
                        if consequent:
                            support_X = frequent_itemsets[frozenset(antecedent)]
                            support_XY = frequent_itemsets[frozenset(itemset)]
                            confidence = support_XY / support_X
                            if confidence >= self.min_confidence:
                                rules.append((antecedent, consequent, confidence))
                                print(f"Rule: {set(antecedent)} -> {set(consequent)}, Confidence: {confidence:.2f}")
        return rules

    def apriori_algorithm(self):
        unique_items = self.get_unique_items()          # generate candidate 1-itemsets
        itemsets = [{item} for item in unique_items]
        frequent_itemsets = {}                          # dictionary to hold the itemsets and their support
        
        # Iteratively find frequent itemsets
        k = 1
        while itemsets:
            print(f"Generating {k}-itemsets...")
            new_itemsets = []
            for itemset in itemsets:
                support = self.calculate_support(itemset)
                if support >= self.min_support:
                    frequent_itemsets[frozenset(itemset)] = support
                    print(f"Itemset: {itemset}, Support: {support:.2f}")
                    # Generate candidate (k+1)-itemsets
                    for other_itemset in itemsets:
                        union_set = itemset.union(other_itemset)
                        if len(union_set) == k + 1 and frozenset(union_set) not in new_itemsets:
                            new_itemsets.append(frozenset(union_set))       # Use frozenset to avoid duplicates
            itemsets = new_itemsets
            k += 1
        return frequent_itemsets
    

if __name__ == "__main__":
    df = pd.read_csv(path)
    ap = apriori(df, 0.05, 0.5)

    # Run the Apriori algorithm
    frequent_itemsets = ap.apriori_algorithm()

    # Print final results
    print("\nFinal frequent itemsets:")
    for itemset, support in frequent_itemsets.items():
        print(f"{set(itemset)}: {support:.2f}")
    
    print("\nRule and confidence:")
    rules = ap.generate_association_rules(frequent_itemsets)