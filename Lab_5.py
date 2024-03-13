import matplotlib.pyplot as plt
import numpy as np
import json

from enum import Enum


class DataType(Enum):
    ARRAY = 'Массив'
    INTERVALS = 'Интервалы'


class Data:
    def __init__(self):
        self.data = {}

    @staticmethod
    def __input_type():
        return input(f"\nВведите тип данных {DataType.ARRAY.value} или {DataType.INTERVALS.value} (1-2): ")

    @staticmethod
    def __get_file_path(file_type):
        if file_type == '1':
            return 'data_array.json'

        elif file_type == '2':
            return 'data_intervals.json'

        else:
            return None

    @staticmethod
    def __get_data_type(data_type):
        if data_type == '1':
            return DataType.ARRAY

        elif data_type == '2':
            return DataType.INTERVALS

        else:
            return None

    @staticmethod
    def __read_file(file_path):
        try:
            with open(file_path, 'r') as file:
                print(f"Файл '{file_path}' успешно найден.")

                return json.load(file)
        except FileNotFoundError:
            print(f"Ошибка: Файл '{file_path}' не найден.")
        except json.JSONDecodeError as e:
            print(f"Ошибка при декодировании JSON в файле '{file_path}': {e}")
        except ValueError as e:
            print(f"Ошибка значения при чтении файла '{file_path}': {e}")
        except Exception as e:
            print(f"Произошла неизвестная ошибка при чтении файла '{file_path}': {e}")

    @staticmethod
    def __write_data(data_type):
        data = {'type': data_type}

        if data_type == DataType.ARRAY:
            data['data'] = [float(x) for x in input("Введите числа через пробел: ").split()]

        elif data_type == DataType.INTERVALS:
            num_values = int(input("\nВведите количество значений: "))
            data['data'] = []

            for value in range(num_values):
                print(f"\nЗначение {value + 1}:")
                start = float(input("Введите начало интервала: "))
                end = float(input("Введите конец интервала: "))
                frequency = float(input("Введите частоту: "))
                data["data"].append({"start": start, "end": end, "frequency": frequency})

        else:
            print("Неизвестный тип данных.")

        return data

    def input_file(self):
        while True:
            input_type = self.__input_type()
            file_type = self.__get_data_type(input_type)
            file_path = self.__get_file_path(input_type)

            if file_type in [type for type in DataType]:
                self.data = self.__read_file(file_path)
                break

            else:
                print("\nНекорректный выбор. Пожалуйста, выберите 1 или 2.")

    def input_keyboard(self=None):
        while True:
            file_type = self.__input_type()
            data_type = self.__get_data_type(file_type)

            if file_type in [type for type in DataType]:
                self.data = self.__write_data(data_type)
                break

            else:
                print("\nНекорректный выбор. Пожалуйста, выберите 1 или 2.")

    def get_data(self=None):
        return self.data


class Statistics:

    def __init__(self):
        self.data = {}

    def set_data(self, data):
        self.data = data

    def __get_data_type(self):
        return self.data['type']

    def __data_type_is_array(self):
        return self.data['type'] == DataType.ARRAY.name

    def __data_type_is_intervals(self):
        return self.data['type'] == DataType.INTERVALS.name

    def __get_data_values(self):
        return self.data['data']

    def __calculate_xi_values(self):
        data_values = self.__get_data_values()

        if self.__data_type_is_array():
            return np.array(data_values)

        if self.__data_type_is_intervals():
            interval_values = [interval for intervals in data_values
                               for interval in (intervals["start"], intervals["end"])]

            return list(set(interval_values))

    def __calculate_average_xi_values(self):
        xi_values = self.__calculate_xi_values

        return [(xi + xi_next) / 2 for xi, xi_next in zip(xi_values[:-1], xi_values[1:])]

    def __calculate_ni_values(self):
        data_values = self.__get_data_values()

        if self.__data_type_is_array():
            return np.array(data_values)

        if self.__data_type_is_intervals():
            return list(np.array([values["frequency"] for values in data_values]))

    def show_data(self=None):
        print(self.data.values())
        data_type = self.data.get('type')
        data_body = np.array(self.data['data'])

        if data_type == DataType.ARRAY.value:
            print("\nСодержимое файла:\n", data_body)

        elif data_type == DataType.INTERVALS.value:
            print("\nСодержимое файла:\n", data_body)

        else:
            print("\nНеизвестный тип данных:", data_type)

    @staticmethod
    def __format_interval(start, end):
        return f"{start} ; {end}"

    def __calculate_variation_series(self):
        data_values = self.__get_data_values()

        if self.__data_type_is_array():
            return sorted(set(data_values))

        if self.__data_type_is_intervals():
            return [self.__format_interval(interval["start"], interval["end"]) for interval in data_values]

    def display_variation_series(self=None):
        variation_series = self.__calculate_variation_series()
        print(variation_series)

    def __calculate_frequency_distribution(self, data):
        variation_series = self.__calculate_variation_series()
        data_values = self.__get_data_values()
        frequencies = []

        if self.__data_type_is_array():
            frequencies = np.array([data_values.count(value) for value in variation_series])

        elif self.__data_type_is_intervals():
            frequencies = np.array([values["frequency"] for values in data_values])

        relative_frequencies = frequencies / sum(frequencies)

        return variation_series, frequencies, relative_frequencies

    def display_frequency_distribution(self=None):
        variation_series, frequencies, relative_frequencies = (self.__calculate_frequency_distribution(self.data))

        print("\nСтатистический ряд частот:")
        for value, freq, rel_freq in zip(variation_series, frequencies, relative_frequencies):
            print(f"Значение: {value}, Частота: {freq}, Относительная частота: {rel_freq}")

    def __plot_histogram(self, data):
        variation_series, frequencies, relative_frequencies = (
            self.__calculate_frequency_distribution(data))

        plt.bar(variation_series, frequencies, width=0.8, align='center', alpha=0.7)
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.title('Статистический ряд')
        plt.show()

    def display_histogram(self=None):
        self.__plot_histogram(self.data)

    def __plot_polygon(self, data):
        variation_series, frequencies, relative_frequencies = (
            self.__calculate_frequency_distribution(data))

        plt.plot(variation_series, relative_frequencies, marker='o')
        plt.xlabel('Значения')
        plt.ylabel('Относительная частота')
        plt.title('Полигон распределения')
        plt.show()

    def display_polygon(self=None):
        self.__plot_polygon(self.data)

    def __calculate_empirical_distribution(self, data):
        variation_series, frequencies, _ = (
            self.__calculate_frequency_distribution(data))

        sum_frequencies = sum(frequencies)

        probability_distribution = frequencies / sum_frequencies
        cumulative_distribution = np.cumsum(frequencies) / sum_frequencies

        return variation_series, probability_distribution, cumulative_distribution

    def display_empirical_distribution(self=None):
        variation_series, probability_distribution, cumulative_distribution = (
            self.__calculate_empirical_distribution(self.data))

        print("\nЭмпирическая функция распределения:")
        for variation, probability, cumulative in zip(variation_series, probability_distribution,
                                                      cumulative_distribution):
            print(f"Значение: {variation},"
                  f"Уникальные значения: {probability},"
                  f"Накопительная функция распределения: {cumulative}")

    def __plot_empirical_distribution(self, data):
        variation_series, probability_distribution, cumulative_distribution = (self.__calculate_empirical_distribution(data))

        plt.plot(variation_series, cumulative_distribution, marker='o')
        plt.xlabel('Значения')
        plt.ylabel('Накопительная функция распределения')
        plt.title('График эмпирической функции распределения')
        plt.show()

    def display_plot_empirical_distribution(self=None):
        self.__plot_empirical_distribution(self.data)

    @staticmethod
    def __text_numerical_characteristics_formulas_array():
        print("\nФормула для среднего значения:")
        print("сумма всех значений / количество значений")

        print("\nФормула для дисперсии:")
        print("сумма квадратов разностей между каждым значением и средним, деленная на количество значений")

        print("\nФормула для стандартного отклонения:")
        print("квадратный корень из дисперсии")

        print("\nФормула для размаха:")
        print("разница между максимальным и минимальным значениями в ряде")

    @staticmethod
    def __text_numerical_characteristics_formulas_intervals():
        print("\nФормула для среднего значения:")
        print("сумма (середина интервала * частота) / общее количество значений")

        print("\nФормула для дисперсии:")
        print("(сумма (квадрат разности среднего значения интервала и среднего значения всего ряда * частота)) "
              "/ общее количество значений")

        print("\nФормула для стандартного отклонения:")
        print("квадратный корень из дисперсии интервалов")

        print("\nФормула для размаха:")
        print("разница между верхним и нижним концами интервала с наибольшей длиной")

    def __calculate_mean(self):
        if self.__data_type_is_array():
            xi_values = self.__calculate_xi_values()
            return np.mean(xi_values)

        if self.__data_type_is_intervals():
            average_xi_values = self.__calculate_average_xi_values()
            ni_values = self.__calculate_ni_values()

            return sum(xi * ni for xi, ni in zip(average_xi_values, ni_values)) / sum(ni_values)

    def __calculate_variance(self):
        ni_values = self.__calculate_ni_values()

        if self.__data_type_is_array():
            xi_values = self.__calculate_xi_values()

            return np.var(xi_values)

        if self.__data_type_is_intervals():
            average_xi_values = self.__calculate_average_xi_values()
            mean = self.__calculate_mean()

            return sum(ni_values[i] * ((average_xi_values[i] - mean) ** 2)
                       for i in range(len(average_xi_values))) / sum(ni_values)

    def __calculate_std_deviation(self):
        if self.__data_type_is_array:
            xi_values = self.__calculate_xi_values()

            return np.std(xi_values)

        if self.__data_type_is_intervals():
            variance = self.__calculate_variance()

            return np.sqrt(variance)

    def __calculate_data_range(self):
        xi_values = self.__calculate_xi_values()
        if self.__data_type_is_array:
            return np.ptp(xi_values)

        if self.__data_type_is_intervals():
            return np.sqrt(xi_values)

    def __calculate_numerical_characteristics(self):
        mean = self.__calculate_mean()
        variance = self.__calculate_variance()
        std_deviation = self.__calculate_std_deviation()
        data_range = self.__calculate_data_range()

        return mean, variance, std_deviation, data_range

    def display_numerical_characteristics(self=None):
        mean, variance, std_deviation, data_range = self.__calculate_numerical_characteristics()
        data_type = self.data['type']
        print("\nЧисловые характеристики выборки:")

        if data_type == DataType.ARRAY.value:
            self.__text_numerical_characteristics_formulas_array()

        if data_type == DataType.INTERVALS.value:
            self.__text_numerical_characteristics_formulas_intervals()

        print("\nСреднее значение (x̄):", mean)
        print("Дисперсия (D):", variance)
        print("Стандартное отклонение (σ):", std_deviation)
        print("Размах (S):", data_range)


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
    data_instance = Data()
    statistics_instance = Statistics()

    data_loaded = False

    while True:
        Menu.display_input()

        choice = input("Выберите пункт меню (0-2): ")

        if choice == '0':
            print("Программа завершена.")
            break
        elif choice == '1':
            data_instance.input_file()
            data_loaded = True
            break
        elif choice == '2':
            data_instance.input_keyboard()
            data_loaded = True
            break
        else:
            print("Некорректный выбор. Пожалуйста, выберите 1 или 2.")

    if data_loaded:
        statistics_instance.set_data(data_instance.get_data())

        while True:
            Menu.display_main()
            choice = input("Выберите пункт меню (0-8): ")

            if choice == '0':
                print("Программа завершена.")
                break
            elif choice == '1':
                statistics_instance.show_data()
            elif choice == '2':
                statistics_instance.display_variation_series()
            elif choice == '3':
                statistics_instance.display_frequency_distribution()
            elif choice == '4':
                statistics_instance.display_histogram()
            elif choice == '5':
                statistics_instance.display_polygon()
            elif choice == '6':
                statistics_instance.display_empirical_distribution()
            elif choice == '7':
                statistics_instance.display_plot_empirical_distribution()
            elif choice == '8':
                statistics_instance.display_numerical_characteristics()
            else:
                print("Некорректный выбор. Пожалуйста, выберите от 0 до 8.")


if __name__ == "__main__":
    main()