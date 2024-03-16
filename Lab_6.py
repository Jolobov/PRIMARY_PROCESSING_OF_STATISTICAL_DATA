import json
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
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

    def calculate_processed_data(self):
        raw_data = self.raw_data

        if self.data_is_array():
            return np.array(raw_data)

        if self.data_is_intervals():
            interval_values = [interval for intervals in raw_data
                               for interval in (intervals["start"], intervals["end"])]
            return list(set(interval_values))

    def calculate_average_processed_data(self):
        return [(interval['start'] + interval['end']) / 2 for interval in self.raw_data]

    def calculate_frequencies(self):
        if self.data_is_array():
            return np.array(self.raw_data())

        if self.data_is_intervals():
            return np.array([interval["frequency"] for interval in self.raw_data])

    @staticmethod
    def format_interval(start, end):
        return f"{start};{end}"

    def calculate_variation_series(self):
        if self.data_is_array():
            return sorted(set(self.raw_data))

        if self.data_is_intervals():
            return [self.format_interval(interval['start'], interval['end']) for interval in self.raw_data]

    def calculate_frequency_distribution(self):
        variation_series = self.calculate_variation_series()
        raw_data = self.raw_data
        frequencies = []

        if self.data_is_array():
            frequencies = np.array([raw_data.count(value) for value in variation_series])

        if self.data_is_intervals():
            frequencies = np.array([values["frequency"] for values in raw_data])

        relative_frequencies = frequencies / sum(frequencies)

        return variation_series, frequencies, relative_frequencies

    def calculate_empirical_distribution(self):
        variation_series, frequencies, _ = self.calculate_frequency_distribution()

        total_frequencies = sum(frequencies)

        probability_distribution = frequencies / total_frequencies
        cumulative_distribution = np.cumsum(frequencies) / total_frequencies

        return variation_series, probability_distribution, cumulative_distribution

    def calculate_mean(self):
        if self.data_is_array():
            processed_data = self.calculate_processed_data()
            return np.mean(processed_data)

        if self.data_is_intervals():
            average_processed_data = self.calculate_average_processed_data()
            frequencies = self.calculate_frequencies()
            return sum(xi * ni for xi, ni in zip(average_processed_data, frequencies)) / sum(frequencies)

    def calculate_variance(self):
        if self.data_is_array():
            processed_data = self.calculate_processed_data()
            return np.var(processed_data)

        if self.data_is_intervals():
            average_processed_data = self.calculate_average_processed_data()
            frequencies = self.calculate_frequencies()
            mean = self.calculate_mean()
            return sum(frequencies[i] * ((average_processed_data[i] - mean) ** 2)
                       for i in range(len(average_processed_data))) / sum(frequencies)

    def calculate_std_deviation(self):
        if self.data_is_array():
            processed_data = self.calculate_processed_data()
            return np.std(processed_data)

        if self.data_is_intervals():
            variance = self.calculate_variance()
            return np.sqrt(variance)

    def calculate_data_range(self):
        if self.data_is_array():
            processed_data = self.calculate_processed_data()
            return np.linspace(min(processed_data),
                               max(processed_data), 100)

        if self.data_is_intervals():
            all_intervals = self.raw_data
            return np.linspace(min(interval['start'] for interval in all_intervals),
                               max(interval['end'] for interval in all_intervals), 100)

    def calculate_numerical_characteristics(self):
        mean = self.calculate_mean()
        variance = self.calculate_variance()
        std_deviation = self.calculate_std_deviation()
        data_range = self.calculate_data_range()
        return mean, variance, std_deviation, data_range

    def calculate_theoretical_probabilities(self):
        if not self.data_is_intervals():
            raise ValueError("\nТеоретические вероятности могут быть вычислены только для интервальных данных.")

        mean, _, std_dev, _ = self.calculate_numerical_characteristics()
        probabilities = []

        for interval in self.raw_data:
            start, end = interval['start'], interval['end']
            p_start = norm.cdf(start, mean, std_dev)
            p_end = norm.cdf(end, mean, std_dev)
            p_interval = p_end - p_start
            probabilities.append(p_interval)

        return probabilities

    def calculate_chi_squared(self):
        observed_frequencies = [interval['frequency'] for interval in self.raw_data]
        expected_frequencies = self.calculate_theoretical_probabilities()
        return sum((obs - exp) ** 2 / exp for obs, exp in zip(observed_frequencies, expected_frequencies))

    def degrees_of_freedom(self):
        number_of_intervals = len(self.raw_data)
        return number_of_intervals - 1 - 2

    def calculate_chi_squared_critical(self, alpha):
        degrees_of_freedom = self.degrees_of_freedom()
        return chi2.ppf(1 - alpha, degrees_of_freedom)

    @staticmethod
    def prompt_for_significance_level():
        while True:
            try:
                alpha = float(input("\nВведите уровень значимости (обычно 0.05): "))
                if 0 < alpha < 1:
                    return alpha
                else:
                    print("\nУровень значимости должен быть числом между 0 и 1 (не включая 0 и 1).")
            except ValueError:
                print("\nНекорректный ввод. Уровень значимости должен быть числом.")


class Visualizer:
    def __init__(self, new_data_processor: DataProcessor):
        self.data_processor = new_data_processor
        pass

    def print_data(self):
        data_type = self.data_processor.data_type
        data_body = self.data_processor.raw_data

        if data_type in (DataType.ARRAY.value, DataType.INTERVALS.value):
            print(f"\nСодержимое файла (тип данных {data_type}):\n", data_body)

        else:
            print("\nНеизвестный тип данных:", data_type)

    def print_variation_series(self):
        variation_series = self.data_processor.calculate_variation_series()
        print(f"\nВариационный ряд:\n", variation_series)

    def print_frequency_distribution(self):
        variation_series, frequencies, relative_frequencies = \
            (self.data_processor.calculate_frequency_distribution())

        print("\nСтатистический ряд частот:")
        for value, freq, rel_freq in zip(variation_series, frequencies, relative_frequencies):
            print(f"Значение: {value}, Частота: {freq}, Относительная частота: {rel_freq}")

    def plot_distribution(self):
        _, frequencies, relative_frequencies = self.data_processor.calculate_frequency_distribution()
        mean, _, std_dev, data_range = self.data_processor.calculate_numerical_characteristics()

        plt.figure(figsize=(10, 6))

        # Построение гистограммы для массива данных
        if self.data_processor.data_is_array():
            plt.hist(self.data_processor.calculate_processed_data(), bins='auto', density=True, alpha=0.7,
                     label='Эмпирическое распределение')

        # Построение гистограммы для интервальных данных
        elif self.data_processor.data_is_intervals():
            # Вычисление центров интервалов
            interval_centers = np.array(
                [(interval['start'] + interval['end']) / 2 for interval in self.data_processor.raw_data])
            # Вычисление ширины интервалов
            widths = [interval['end'] - interval['start'] for interval in self.data_processor.raw_data]
            # Построение гистограммы
            plt.bar(interval_centers, relative_frequencies, width=widths, alpha=0.7, label='Эмпирическое распределение')

        # Построение графика плотности нормального распределения
        x_values = np.linspace(min(data_range), max(data_range), 1000)
        y_values = norm.pdf(x_values, mean, std_dev)
        plt.plot(x_values, y_values, label='Нормальное распределение', color='red')

        plt.legend()
        plt.title('Гистограмма и плотность нормального распределения')
        plt.xlabel('Значение')
        plt.ylabel('Плотность вероятности')
        plt.show()

    def print_distribution_formula(self):
        _, _, std_dev, _ = self.data_processor.calculate_numerical_characteristics()
        print("\nФормула плотности нормального распределения с оцененными параметрами:")
        print(f"f(x) = 1 / (σ*√(2π)) * e^(-(x - a*)^2 / (2σ*^2))")
        print(f"где a* = среднее выборки, σ* = {std_dev}")

    def print_theoretical_probabilities(self):
        if self.data_processor.data_is_intervals():
            probabilities = self.data_processor.calculate_theoretical_probabilities()
            for i, probability in enumerate(probabilities):
                print(f"Интервал {i + 1}: теоретическая вероятность = {probability:.4f}")
        else:
            print("\nДанные не представляют интервалы.")
            print("Необходимы интервальные данные для расчета теоретических вероятностей.")

    def print_chi_squared(self):
        chi_squared = self.data_processor.calculate_chi_squared()
        print(f"\nФормула для вычисления наблюдаемого хи-квадрат: X^2 = Σ((O_i - E_i)^2 / E_i)")
        print(f"Наблюдаемое значение хи-квадрат: {chi_squared:.4f}")

    def print_significance_and_degrees_of_freedom(self):
        alpha = self.data_processor.prompt_for_significance_level()
        df = self.data_processor.degrees_of_freedom()
        print("\nФормула для расчета числа степеней свободы: df = n - p - 1")
        print("где df - число степеней свободы, n - количество интервалов")
        print("    p - количество оцененных параметров (среднее и стандартное отклонение).")
        print(f"\nЗаданный уровень значимости: {alpha}")
        print(f"Число степеней свободы: {df}")

    def print_chi_squared_critical_value(self):
        alpha = self.data_processor.prompt_for_significance_level()
        chi_squared_critical = self.data_processor.calculate_chi_squared_critical(alpha)
        print(f"\nФормула для вычисления критического значения хи-квадрат")
        print(f"  на уровне значимости α и числе степеней свободы k:")
        print(f"\nχ²крит = F⁻¹(1 - α, k)")
        print(f"\nгде F⁻¹ - обратная функция распределения хи-квадрат,")
        print(f"    α - уровень значимости, k - число степеней свободы.")
        print(f"\nКритическое значение хи-квадрат при уровне значимости {alpha}: {chi_squared_critical:.4f}")

    def print_chi_squared_critical_table(self):
        df = self.data_processor.degrees_of_freedom()
        alphas = [0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        print(f"\nТаблица критических значений хи-квадрат для {df} степеней свободы:\n")
        print(f"{'Уровень значимости':<20}{'Критическое значение':<20}")
        for alpha in alphas:
            chi_squared_critical = chi2.ppf(1 - alpha, df)
            print(f"{alpha:<20}{chi_squared_critical:<20.4f}")

    def print_chi_squared_test_result(self):
        alpha = self.data_processor.prompt_for_significance_level()
        chi_squared = self.data_processor.calculate_chi_squared()
        chi_squared_critical = self.data_processor.calculate_chi_squared_critical(alpha)
        print(f"Наблюдаемое значение хи-квадрат: {chi_squared:.4f}")
        print(f"Критическое значение хи-квадрат для уровня значимости {alpha}: {chi_squared_critical:.4f}")

        if chi_squared < chi_squared_critical:
            print("Нулевая гипотеза принимается.")
        else:
            print("Нулевая гипотеза отвергается.")


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
        print("4. Построить гистограмму и плотность нормального распределения")
        print("5. Вывести формулу плотности нормального распределения")
        print("6. Вывести теоретические вероятности попадания значений (нормальное распределение)")
        print("7. Вывести результаты теста хи-квадрат")
        print("8. Вывести уровень значимости и число степеней свободы")
        print("9. Вывести критическое значение хи-квадрат")
        print("10. Вывести таблицу критических значений хи-квадрат")
        print("11. Сравнить наблюдаемое и критическое значение хи-квадрат и вывести результат")
        print("0. Выход")


def main():
    data_manager = DataManager()
    json_data = data_manager.input_data()

    data_processor = DataProcessor(json_data)
    data_visualizer = Visualizer(data_processor)

    while True:
        Menu.display_main()
        choice = input("Выберите пункт меню (0-11): ")

        if choice == '0':
            print("Программа завершена.")
            break
        elif choice == '1':
            data_visualizer.print_data()
        elif choice == '2':
            data_visualizer.print_variation_series()
        elif choice == '3':
            data_visualizer.print_frequency_distribution()
        elif choice == '4':
            data_visualizer.plot_distribution()
        elif choice == '5':
            data_visualizer.print_distribution_formula()
        elif choice == '6':
            data_visualizer.print_theoretical_probabilities()
        elif choice == '7':
            data_visualizer.print_chi_squared()
        elif choice == '8':
            data_visualizer.print_significance_and_degrees_of_freedom()
        elif choice == '9':
            data_visualizer.print_chi_squared_critical_value()
        elif choice == '10':
            data_visualizer.print_chi_squared_critical_table()
        elif choice == '11':
            data_visualizer.print_chi_squared_test_result()
        else:
            print("Некорректный выбор. Пожалуйста, выберите от 0 до 11.")


if __name__ == "__main__":
    main()
