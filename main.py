import pattern_recognition as pr
import tema as tm

# Global variables
data_file = "test_data.tsv"
pattern_file = "test_pattern.tsv"
tema_file = "test_tema.tsv"

data = pr.read_data(data_file)
#pattern = pr.read_data("test_pattern.tsv")
#print(pr.recognize_pattern(data, pattern, abs_weight=0.5))

pred = tm.predict_next_tema(data, 4, 2)
tm.ema_to_file(data, 5)
print(pred)