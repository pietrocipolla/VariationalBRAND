from matplotlib import pyplot as plt

import time
class Record:
    def __init__(self, function_name: str, elapsed_time):
        self.function_name = function_name
        self.elapsed_time = elapsed_time

#HOW TO USE:
# tic = TimeTracker.start()
# TimeTracker.stop_and_save('load_data', tic)

class TimeTracker:
    records = []

    @staticmethod
    def start():
        tic = time.perf_counter()
        return tic

    @staticmethod
    def stop():
        toc = time.perf_counter()
        return toc

    @staticmethod
    def save_record(function_name, toc, tic):
        elapsed_time = toc-tic
        TimeTracker.records.append(Record(function_name, elapsed_time))

    @staticmethod
    def stop_and_save(function_name: str, tic):
        toc = TimeTracker.stop()
        TimeTracker.save_record(function_name, toc, tic)

    @staticmethod
    def test_print():
        print(TimeTracker.records[0].function_name, TimeTracker.records[0].elapsed_time)

    @staticmethod
    def get_performance():
        main_time = ''
        for record in TimeTracker.records:
            print(record.function_name, record.elapsed_time)
            if(record.function_name == 'main'):
                main_time = str(record.function_name) + ': ' + str(record.elapsed_time)

        return main_time

    @staticmethod
    def plot_performance():
        for record in TimeTracker.records:
            print(record.function_name, record.elapsed_time)
            plt.scatter(record.function_name, record.elapsed_time)

        plt.xlabel('function_name')
        plt.ylabel('elapsed_time')
        plt.title('Performance')
        # plt.show()
        plt.savefig('performace.png')
        plt.close()

    @staticmethod
    def plot_main_performance():
        for record in TimeTracker.records:
            function_name = record.function_name
            if ('load_data' in function_name) or ('get_training_set' in function_name) or ('calculate_robust_parameters' in function_name) or \
                ('specify_user_input' in function_name) or ('set_hyperparameters' in function_name) or \
                ('cavi' in function_name) or ('generate_induced_partition' in function_name) or ('generate_elbo_plot' in function_name):
                print(record.function_name, record.elapsed_time)
            plt.scatter(record.function_name, record.elapsed_time)

        plt.xlabel('function_name')
        plt.ylabel('elapsed_time')
        plt.title('Performance Main')
        # plt.show()
        plt.savefig('performace-main.png')
        plt.close()


    @staticmethod
    def plot_update_parameters_performance():
        for record in TimeTracker.records:
            if 'update_NIW_' not in record.function_name:
                print(record.function_name, record.elapsed_time)
                plt.scatter(record.function_name, record.elapsed_time)

        plt.xlabel('function_name')
        plt.ylabel('elapsed_time')
        plt.title('Performance-update_parameters')
        plt.savefig('performance-update_parameters.png')
        plt.close()

        for record in TimeTracker.records:
            if 'update_NIW_' in record.function_name:
                print(record.function_name, record.elapsed_time)
                plt.scatter(record.function_name, record.elapsed_time)


        plt.xlabel('function_name')
        plt.ylabel('elapsed_time')
        plt.title('Performance_update_parameters-NIW')
        # plt.show()
        plt.savefig('performance-update_parameters-NIW.png')
        plt.close()

    @staticmethod
    def plot_elbo_calculator_performance():
        for record in TimeTracker.records:
            if 'e_' in record.function_name:
                print(record.function_name, record.elapsed_time)
                plt.scatter(record.function_name, record.elapsed_time)

        plt.xlabel('function_name')
        plt.ylabel('elapsed_time')
        plt.title('Performance-elbo_calculator')
        plt.savefig('performance-elbo_calculator.png')
        plt.close()

