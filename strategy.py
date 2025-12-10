import prediction as pr


def buy_stock(data: list[float], tema: list[float], pattern: list[float], length: int = 5):
    """Calculates when to buy stock"""
    
    buy = pr.read_data("stock_aquired.tsv")
    sell = pr.read_data("stock_sold.tsv")
    
    if(len(buy) != len(sell)):
        print("Already bought stocks")
        return
    
    for i in range(length):
        if(data[i - length ] < tema[i - length]):
            print("Tema is bigger then data")
            return
    
    if(pattern[-1] < data[-1]):
        print("Pattern is smaller then data")
        return
    
    print(f"Bying stock at {data[-1]}€")
    with open("stock_aquired.tsv", "a") as f:
        f.write(data[-1])

def sell_stock(data: list[float], tema: list[float], pattern: list[float], length: int = 5):
    """Calculets when to sell stock"""
    
    buy = pr.read_data("stock_aquired.tsv")
    sell = pr.read_data("stock_sold.tsv")
    
    if(len(buy) == len(sell)):
        print("No stocks to sell")
        return
    
    for i in range(length):
        if(data[i - length ] > tema[i - length]):
            print("Tema is smaller then data")
            return
    
    if(pattern[-1] > data[-1]):
        print("Pattern is bigger then data")
        return
    
    print(f"Selling stock at {data[-1]}€")
    with open("stock_sold.tsv", "a") as f:
        f.write(data[-1])