import json
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


class DataType(Enum):
    ARRAY = "ARRAY"
    INTERVALS = "INTERVALS"


class DataManager:
    def __init__(self):
        pass

    def input_data(self):
        print("\nВыберите метод ввода данных:")
        print("1 - Ввод данных с клавиатуры")
        print("2 - Ввод данных из файла")
        print("0 - Выход")
        choice = input("Ваш выбор: ")

        if choice == '1':
            return self.input_data_keyboard()
        elif choice == '2':
            file_path = input("Введите путь к файлу: ")
            return self.read_data_from_file(file_path)
        elif choice == '0':
            print("Программа завершена.")
            return
        else:
            print("Некорректный выбор. Попробуйте снова.")
            return self.input_data()

    def input_data_keyboard(self):
        print("\nВыберите тип данных:")
        print(f"1 - {DataType.ARRAY.value}")
        print(f"2 - {DataType.INTERVALS.value}")
        print("0 - Назад")
        choice = input("Ваш выбор: ")

        json_data = {'type': {}, 'data': {}}

        if choice == '1':
            json_data['type'] = DataType.ARRAY.value
            json_data['data'] = list(map(float, input("\nВведите числа через пробел: ").split()))
            return json_data
        elif choice == '2':
            num_values = int(input("\nВведите количество интервалов: "))
            json_data['type'] = DataType.INTERVALS.value
            json_data['data'] = []
            for value in range(num_values):
                print(f"\nЗначение {value + 1}:")
                json_data["data"].append({"start": float(input("Введите начало интервала: ")),
                                          "end": float(input("Введите конец интервала: ")),
                                          "frequency": float(input("Введите частоту: "))})
            return json_data
        elif choice == '0':
            return self.input_data()
        else:
            print("Некорректный выбор. Попробуйте снова.")
            return self.input_data_keyboard()

    def read_data_from_file(self, file_path):
        try:
            with open(file_path, 'r') as file:
                print(f"Файл {file_path} успешно найден")
                return json.load(file)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file_path}' не найден.")
            return self.input_data()
        except json.JSONDecodeError as e:
            print(f"Ошибка при декодировании JSON в файле '{file_path}': {e}")
            return self.input_data()
        except ValueError as e:
            print(f"Ошибка значения при чтении файла '{file_path}': {e}")
            return self.input_data()
        except Exception as e:
            print(f"Произошла неизвестная ошибка при чтении файла '{file_path}': {e}")
            return self.input_data()


class DataProcessor:
    def __init__(self, json_data):
        self.data_type = json_data['type']
        self.raw_data = json_data['data']

    def data_is_array(self):
        return self.data_type == DataType.ARRAY.value

    def data_is_intervals(self):
        return self.data_type == DataType.INTERVALS.value

    def processed_data(self):
        raw_data = self.raw_data

        if self.data_is_array():
            return np.array(raw_data)

        if self.data_is_intervals():
            interval_values = [interval for intervals in raw_data
                               for interval in (intervals["start"], intervals["end"])]
            return list(set(interval_values))

    def average_processed_data(self):
        raw_data = self.raw_data

        return [(xi + xi_next) / 2 for xi, xi_next in zip(raw_data[:-1], raw_data[1:])]

    def frequencies(self):
        raw_data = self.raw_data()

        if self.data_is_array():
            return np.array(raw_data)

        if self.data_is_intervals():
            return list(np.array([values["frequency"] for values in raw_data]))

    @staticmethod
    def format_interval(start, end):
        return f"{start};{end}"

    def variation_series(self):
        raw_data = self.raw_data

        if self.data_is_array():
            return sorted(set(raw_data))

        if self.data_is_intervals():
            return [self.format_interval(interval['start'], interval['end']) for interval in raw_data]

    def frequency_distribution(self):
        variation_series = self.variation_series()
        raw_data = self.raw_data
        frequencies = []

        if self.data_is_array():
            frequencies = np.array([raw_data.count(value) for value in variation_series])

        if self.data_is_intervals():
            frequencies = np.array([values["frequency"] for values in raw_data])

        relative_frequencies = frequencies / sum(frequencies)

        return variation_series, frequencies, relative_frequencies

    def empirical_distribution(self):
        variation_series, frequencies, _ = self.frequency_distribution()

        total_frequencies = sum(frequencies)

        probability_distribution = frequencies / total_frequencies
        cumulative_distribution = np.cumsum(frequencies) / total_frequencies

        return variation_series, probability_distribution, cumulative_distribution

    def mean(self):
        if self.data_is_array():
            processed_data = self.processed_data()
            return np.mean(processed_data)

        if self.data_is_intervals():
            average_processed_data = self.average_processed_data()
            frequencies = self.frequencies()
            return sum(xi * ni for xi, ni in zip(average_processed_data, frequencies)) / sum(frequencies)

    def variance(self):
        if self.data_is_array():
            processed_data = self.processed_data()
            return np.var(processed_data)

        if self.data_is_intervals():
            average_processed_data = self.average_processed_data()
            frequencies = self.frequencies()
            mean = self.mean()
            return sum(frequencies[i] * ((average_processed_data[i] - mean) ** 2)
                       for i in range(len(average_processed_data))) / sum(frequencies)

    def std_deviation(self):
        if self.data_is_array():
            processed_data = self.processed_data()
            return np.std(processed_data)

        if self.data_is_intervals():
            variance = self.variance()
            return np.sqrt(variance)

    def data_range(self):
        if self.data_is_array():
            processed_data = self.processed_data()
            return np.linspace(min(processed_data),
                               max(processed_data), 100)

        if self.data_is_intervals():
            all_intervals = self.raw_data
            return np.linspace(min(interval['start'] for interval in all_intervals),
                               max(interval['end'] for interval in all_intervals), 100)

    def numerical_characteristics(self):
        mean = self.mean()
        variance = self.variance()
        std_deviation = self.std_deviation()
        data_range = self.data_range()
        return mean, variance, std_deviation, data_range

    ## todo Дописать методы c 7


class Visualizer:
    def __init__(self, new_data_processor: DataProcessor):
        self.data_processor = new_data_processor
        pass

    ## todo Настроить Визуализацию данных

    def show_data(self):
        data_type = self.data_processor.data_type
        data_body = self.data_processor.raw_data

        print(data_type)
        print(data_body)

        if data_type in (DataType.ARRAY.value, DataType.INTERVALS.value):
            print(f"\nСодержимое файла (тип данных {data_type}):\n", data_body)

        else:
            print("\nНеизвестный тип данных:", data_type)

    def variation_series(self):
        variation_series = self.data_processor.variation_series()
        print(f"\nВариационный ряд:\n", variation_series)

    def frequency_distribution(self):
        variation_series, frequencies, relative_frequencies = \
            (self.data_processor.frequency_distribution())

        print("\nСтатистический ряд частот:")
        for value, freq, rel_freq in zip(variation_series, frequencies, relative_frequencies):
            print(f"Значение: {value}, Частота: {freq}, Относительная частота: {rel_freq}")

    def plot_distribution(self):
        _, frequencies, relative_frequencies = self.data_processor.frequency_distribution()
        mean, _, std_dev, data_range = self.data_processor.numerical_characteristics()

        plt.figure(figsize=(10, 6))

        # Построение гистограммы относительных частот
        # Если данные представлены массивом значений
        if self.data_processor.data_is_array():
            plt.hist(self.data_processor.processed_data(), bins='auto', density=True, alpha=0.7,
                     label='Эмпирическое распределение')
        # Если данные представлены интервалами
        elif self.data_processor.data_is_intervals():
            interval_centers = np.array(
                [(interval['start'] + interval['end']) / 2 for interval in self.data_processor.raw_data])
            plt.bar(interval_centers, relative_frequencies, width=np.diff(data_range).mean(), alpha=0.7,
                    label='Эмпирическое распределение')

        # Построение графика плотности нормального распределения
        plt.plot(data_range, norm.pdf(data_range, mean, std_dev), label='Нормальное распределение', color='red')
        plt.legend()
        plt.title('Гистограмма и плотность нормального распределения')
        plt.xlabel('Значение')
        plt.ylabel('Плотность вероятности')
        plt.show()

    def print_distribution_formula(self):
        _, _, std_dev, _ = self.data_processor.numerical_characteristics()
        print("\nФормула плотности нормального распределения с оцененными параметрами:")
        print(f"f(x) = 1 / (σ*√(2π)) * e^(-(x - a*)^2 / (2σ*^2))")
        print(f"где a* = среднее выборки, σ* = {std_dev}")


## todo Переписать меню

class Menu:
    @staticmethod
    def display_input():
        print("\nМеню:")
        print("1. Ввод из файла")
        print("2. Ввод с клавиатуры")
        print("0. Выход")

    @staticmethod
    def display_main():
        print("\nМеню:")
        print("1. Вывести содержимое файла")
        print("2. Вывести вариационный ряд")
        print("3. Вывести статистический ряд частот и относительных частот")
        print("4. Построить гистограмму распределения")
        print("5. Построить полигон распределения")
        print("6. Вывести эмпирическую функцию распределения")
        print("7. Вывести график эмпирической функции распределения")
        print("8. Вывести числовые характеристики выборки")
        print("0. Выход")


def main():
    data_manager = DataManager()
    json_data = data_manager.input_data()

    data_processor = DataProcessor(json_data)
    data_visualizer = Visualizer(data_processor)

    while True:
        Menu.display_main()
        choice = input("Выберите пункт меню (0-8): ")

        if choice == '0':
            print("Программа завершена.")
            break
        elif choice == '1':
            data_visualizer.show_data()
        elif choice == '2':
            data_visualizer.variation_series()
        elif choice == '3':
            data_visualizer.frequency_distribution()
        elif choice == '4':
            data_visualizer.plot_distribution()
        elif choice == '5':
            data_visualizer.print_distribution_formula()
        else:
            print("Некорректный выбор. Пожалуйста, выберите от 0 до 4.")


if __name__ == "__main__":
    main()
