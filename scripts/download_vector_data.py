import urllib.request

f = open("classes.txt", "r")
classes = f.readlines()
f.close()


def download():
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for c in classes:
        clean_class = c.strip()
        path = base + clean_class + '.npy'
        print(path)
        urllib.request.urlretrieve(path, 'data/' + clean_class + '.npy')


if __name__ == "__main__":
    download()
