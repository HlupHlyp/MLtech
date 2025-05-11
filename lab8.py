import gradio as gr
import pandas as pd
import graphviz
from sklearn.tree import DecisionTreeRegressor
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def load_data():
    '''
    Загрузка данных
    '''
    data = pd.read_csv('Student_performance_data.csv', sep=",")
    return data

def preprocess_data(data_in):
    '''
    Масштабирование признаков, функция возвращает X и y для кросс-валидации
    '''
    required_data = data_in[["GPA", "StudyTimeWeekly", "Absences", "Tutoring", 
                         "ParentalSupport"]].copy()

    # Числовые колонки для масштабирования
    data_X_train, data_X_test, data_y_train, data_y_test = train_test_split(
    required_data[["StudyTimeWeekly", "Absences", "Tutoring", 
                         "ParentalSupport"]], required_data["GPA"], test_size=0.2, random_state=1)
    return data_X_train, data_X_test, data_y_train, data_y_test

data = load_data()
data_X_train, data_X_test, data_y_train, data_y_test = preprocess_data(data)

def get_png_tree(tree_model_param, feature_names_param):
    dot_data = StringIO()
    export_graphviz(tree_model_param, out_file=dot_data, feature_names=feature_names_param,
                    filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data.getvalue())
    return graph.render(format='png', view=False)

def knn(max_depth, min_samples_leaf, max_features, criterion):
    '''
    Входы и выходы функции соединены с компонентами в интерфейсе
    '''
    tree_model = DecisionTreeRegressor(max_depth=max_depth, 
                                   random_state=1, 
                                   min_samples_leaf=min_samples_leaf, 
                                   criterion=criterion, 
                                    max_features=max_features)
    tree_model.fit(data_X_train, data_y_train)
    MAE = mean_absolute_error(data_y_test, tree_model.predict(data_X_test))
    MSE = mean_squared_error(data_y_test, tree_model.predict(data_X_test))
    graph = get_png_tree(tree_model, data_X_train.columns)
    return {'MAE':MAE,'MSE':MSE}, graph



#Входные компоненты
max_depth = gr.Slider(minimum=1, maximum=10, step=1, value=4, label='Макс. глубина дерева')
min_samples_leaf = gr.Slider(minimum=2, maximum=10, step=1, value=5, label='Мин. образцов на лист') 
max_features = gr.Dropdown(choices=['sqrt', 'log2'], 
                           label='Макс. кол-во признаков при разбиении узла')
criterion = gr.Dropdown(choices=["squared_error", "friedman_mse", "absolute_error", "poisson"], 
                               label='Критерий оценки')

#Выходные компоненты
metrics = gr.Label(label='Метрики', show_heading=False)
decision_tree_schema = gr.Image(label='Схема дерева решений')

iface = gr.Interface(
  fn=knn, 
  inputs=[max_depth, min_samples_leaf, max_features, criterion], 
  outputs=[metrics, decision_tree_schema],
  title='Метод ближайших соседей')
iface.launch()
