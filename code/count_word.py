def load_file_txt(source):
    file = open(source, 'r')
    text = file.read()
    return text.split("\n")

def load_data(name='id.albaqarah.cut.txt'):
    data = load_file_txt(name)
    expected = "".join([str(i.split(",")[0]) for i in data])
    return data, expected

if __name__ == '__main__':
    sent, expected = load_data('data/sintesis.extreme1.txt')
    import pdb
    pdb.set_trace()