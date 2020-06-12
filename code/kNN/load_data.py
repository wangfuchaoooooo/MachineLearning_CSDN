import pandas as pd  # 导入pandas库
import numpy as np
import string


def load_data(nums=0):
    movies_data = [('CM', 3, 104, '爱情片'), ('HRD', 2, 100, '爱情片'), ('BW', 1, 81, '爱情片'),
                   ('KL', 101, 10, '动作片'), ('RS', 99, 5, '动作片'), ('A', 98, 2, '动作片'),
                   ]
    for _ in range(nums):
        movies_name_len = np.random.randint(2, 6)
        movies_name = ''
        for n in range(movies_name_len):
            movies_name = movies_name + string.ascii_letters[np.random.randint(1, 26)]
        fights = np.random.randint(1, 200)
        kisses = np.random.randint(1, 200)
        if fights >= kisses:
            category = '动作片'
        else:
            category = '爱情片'
        movies_data.append((movies_name, fights, kisses, category))
    df = pd.DataFrame(data=movies_data, columns=['电影名称', '打斗镜头', '接吻镜头', '电影类别'])
    df.to_csv('ope/movies_data.csv', index=None)
    return df


# if __name__ == '__main__':
#     d = load_data(100)
#     print(d)
