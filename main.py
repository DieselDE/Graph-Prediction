import time
import prediction as pr
import strategy as sta
import data_collection as dc


if __name__ == "__main__":
        
    i = 0
    
    while(True):
        price = dc.get_bitcoin_price()
        if price is not None:
            print(f"The current price of Bitcoin is: €{price}")
            with open("test_data.tsv", "a") as f:
                f.write(str(price) + "\n")
            i += 1
            print(f"We are at {i} items in file")
            
            # Read your Bitcoin price data
            data = pr.read_data("test_data.tsv")

            # Get prediction
            prediction = []
            if i % 5 == 0:
                prediction = pr.find_pattern(data, 20, 5, 1)

            print(f"Pattern prediction: €{prediction}")
            with open("pr_pattern.tsv", "a") as g:
                for j in range(len(prediction)):
                    g.write(str(round(prediction[j], 2)) + "\n")
        else:
            print("Could not retrieve Bitcoin price.")
        
        tema = pr.tema(data, 10)
        with open("pr_tema.tsv","a") as h:
            h.write(str(round(tema[-1], 2)) + "\n")
        
        pattern = pr.read_data("pr_pattern.tsv")
        
        sta.buy_stock(data, tema, pattern)
        sta.sell_stock(data, tema, pattern)
        
        time.sleep(60)