#Bước 1: Nhập thư viện và tải tập dữ liệu
import pandas as pd
import numpy as np

df = pd.read_csv('dataprocessing/Titanic-Dataset.csv')
df.info()
df.head()
# Bước 2: Kiểm tra các hàng trùng lặp
# df.duplicated(): Trả về một chuỗi boolean cho biết các hàng trùng lặp.
df.duplicated()
'''Bước 3: Xác định kiểu dữ liệu cột
Danh sách hiểu với thuộc tính .dtype để tách biệt các cột phân loại và số.
object dtype: Thường được sử dụng cho văn bản hoặc dữ liệu phân loại.'''

cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']

print('Categorical columns:', cat_col)
print('Numerical columns:', num_col)

'''Bước 4: Đếm các giá trị duy nhất trong các cột phân loại
df[cat_col].nunique(): Trả về số lượng giá trị duy nhất trên mỗi cột.'''


df[cat_col].nunique()

'''Bước 5: Tính các giá trị còn thiếu theo tỷ lệ phần trăm
df.isnull(): Phát hiện các giá trị bị thiếu, trả về DataFrame boolean.
Tổng bị thiếu trên các cột, chuẩn hóa với tổng số hàng và nhân với 100.'''

round((df.isnull().sum() / df.shape[0]) * 100, 2)

'''Bước 6: Bỏ các cột không liên quan hoặc thiếu nhiều dữ liệu
df.drop(columns=[]): Thả các cột được chỉ định khỏi DataFrame.
df.dropna(subset=[]): Xóa các hàng trong đó các cột được chỉ định bị thiếu giá trị.
fillna(): Điền các giá trị còn thiếu bằng giá trị được chỉ định (ví dụ: trung bình).'''

df1 = df.drop(columns=['Name', 'Ticket', 'Cabin'])
df1.dropna(subset=['Embarked'], inplace=True)
df1['Age'] = df1['Age'].fillna(df1['Age'].mean())

'''Bước 7: Phát hiện ngoại lệ với biểu đồ hộp
matplotlib.pyplot.boxplot(): Hiển thị sự phân phối dữ liệu, làm nổi bật giá trị trung bình, phần tư và giá trị ngoại lệ.
plt.show(): Hiển thị biểu đồ.'''

import matplotlib.pyplot as plt

plt.boxplot(df1['Age'], vert=False)
plt.ylabel('Variable')
plt.xlabel('Age')
plt.title('Box Plot')
plt.show()

'''Bước 8: Tính toán ranh giới ngoại lệ và loại bỏ chúng
Tính giá trị trung bình và độ lệch chuẩn (std) bằng cách sử dụng df['Age'].mean() và df['Age'].std().
Xác định giới hạn là giá trị trung bình ± 2 * std để phát hiện ngoại lệ.
Lọc các hàng DataFrame trong giới hạn bằng cách sử dụng lập chỉ mục Boolean.'''

mean = df1['Age'].mean()
std = df1['Age'].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

df2 = df1[(df1['Age'] >= lower_bound) & (df1['Age'] <= upper_bound)]

'''Bước 9: Gán lại dữ liệu bị thiếu nếu có
fillna() được áp dụng lại trên dữ liệu đã lọc để xử lý mọi giá trị còn thiếu còn lại.'''

df3 = df2.fillna(df2['Age'].mean())
df3.isnull().sum()

'''Bước 10: Tính toán lại giới hạn ngoại lệ và loại bỏ các ngoại lệ khỏi dữ liệu cập nhật
mean = df3['Tuổi'].mean(): Tính giá trị trung bình (trung bình) của cột Tuổi trong Khung dữ liệu df3.
std = df3['Tuổi'].std(): Tính toán độ lệch chuẩn (chênh lệch hoặc biến đổi) của cột Tuổi trong df3.
lower_bound = mean - 2 * std: Xác định giới hạn dưới cho các giá trị Tuổi chấp nhận được, được đặt dưới hai độ lệch chuẩn dưới giá trị trung bình.
upper_bound = mean + 2 * std: Xác định giới hạn trên cho các giá trị Tuổi được chấp nhận, được đặt dưới hai độ lệch chuẩn trên giá trị trung bình.
df4 = df3[(df3['Tuổi'] >= lower_bound) & (df3['Tuổi'] <= upper_bound)]: Tạo DataFrame df4 mới bằng cách chỉ chọn các hàng có giá trị Độ tuổi nằm giữa giới hạn dưới và giới hạn trên, loại bỏ hiệu quả các độ tuổi ngoại lệ nằm ngoài phạm vi này.'''

mean = df3['Age'].mean()
std = df3['Age'].std()

lower_bound = mean - 2 * std
upper_bound = mean + 2 * std

print('Lower Bound :', lower_bound)
print('Upper Bound :', upper_bound)

df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]
'''Bước 11: Xác thực và xác minh dữ liệu
Xác thực và xác minh dữ liệu liên quan đến việc đảm bảo rằng dữ liệu chính xác và nhất quán bằng cách so sánh dữ liệu với các nguồn bên ngoài hoặc kiến thức chuyên môn.

Đối với dự đoán machine learning, chúng tôi tách biệt các tính năng độc lập và mục tiêu
Ở đây chúng ta sẽ xem xét 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' và 'Embarked' là các tính năng độc lập.
Sống sót dưới dạng biến mục tiêu vì PassengerId sẽ không ảnh hưởng đến tỷ lệ sống sót'''
X = df3[['Pclass','Sex','Age', 'SibSp','Parch','Fare','Embarked']]
Y = df3['Survived']
'''Bước 12: Định dạng dữ liệu
Định dạng dữ liệu liên quan đến việc chuyển đổi dữ liệu thành định dạng hoặc cấu trúc tiêu chuẩn có thể dễ dàng xử lý bằng các thuật toán hoặc mô hình được sử dụng để phân tích. Ở đây chúng ta sẽ thảo luận về các kỹ thuật định dạng dữ liệu thường được sử dụng, tức là Mở rộng quy mô và Chuẩn hóa.

1. Tỷ lệ tối thiểu-tối đa: Mở rộng quy mô liên quan đến việc chuyển đổi giá trị của các tính năng thành một phạm vi cụ thể. Tỷ lệ tối thiểu-tối đa thay đổi tỷ lệ các giá trị đến một phạm vi cụ thể, thường là từ 0 đến 1. Nó giữ nguyên phân phối ban đầu và đảm bảo rằng giá trị nhỏ nhất ánh xạ đến 0 và giá trị lớn nhất ánh xạ đến 1.'''
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

num_col_ = [col for col in X.columns if X[col].dtype != 'object']
x1 = X
x1[num_col_] = scaler.fit_transform(x1[num_col_])
x1.head()