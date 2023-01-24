import numpy as np
import matplotlib.pyplot as plt

class Experiment() :

    def __init__(self, simulator, exp_type="xor"):
        np.random.seed(None)
        self.simulator = simulator
        self.exp_type = exp_type

    def run(self, run_num=10, isit_train=True):
        """

        :param run_num:
        :param isit_train:
        :return:
            1st : is experiment is well done
            2st : exception ment
        """

        self.make_env()

        if self.intput_output_check() == False: return -1
        input_neuron_set = self.simulator.get_type_neurons(0, self.input_num)
        output_neuron_set = self.simulator.get_type_neurons(2, self.output_num)
        feedback_neuron_set = self.simulator.get_type_neurons(3, self.feedback_num)

        for run_cnt in range(run_num):

            # Make Data
            input_data, output_data, feedback_data = self.whole_data[np.random.choice(range(self.whole_data_len), 1)[0]]
            input_state = self.state_map(input_neuron_set, input_data)
            output_state = self.state_map(output_neuron_set, output_data)
            feedback_state = self.state_map(feedback_neuron_set, feedback_data)

            output_state_space = self.state_map(output_neuron_set, self.output_data_space)

            input_input = self.state_dict(input_neuron_set, input_state)
            input_output = self.state_dict(output_neuron_set, output_state)

            # Simulate
            self.simulator.init_node()

            self.debug_neuron_set(output_neuron_set)

            self.simulator.input(input_input)
            for i in range(self.run_max_num):
                self.simulator.sampling()
                if isit_train == True: self.simulator.input(input_output)
                self.simulator.training_save()


            # Validation
            output_probability = self.get_debug_output_probability(output_neuron_set)

            score_list = np.zeros(len(output_state_space))
            for i in range(output_neuron_set) :
                score_list += np.exp(output_probability[i][output_neuron_set[i]])

            score =1
            print(score_list)
            for i in range(self.output_num):
                score*= 0


            self.score_history.append(score)

            self.simulator.training_by_value(score*1000)

        return 0

    def view_score_history(self):
        # Data for plotting
        t = np.arange(0, len(self.score_history))

        fig, ax = plt.subplots()
        ax.plot(t, self.score_history)

        ax.set(xlabel='epoch', ylabel='score',
               title='score board')
        #ax.grid()
        plt.show()

    def make_env(self):

        self.score_history = list()

        if self.exp_type == "xor" :
            self.input_num = 2
            self.output_num = 1
            self.feedback_num = 0
            self.run_max_num = 200

            #encoding
            self.encoding_type = "0to1"
            self.encoding_list = np.array([0,0.5])

            self.whole_data = list()

            for x in range(2) :
                for y in range(2) :
                    training_data = self.encoding_list[[x,y]]
                    test_data = self.encoding_list[[x^y]]
                    feedback_data = self.encoding_list[[]]
                    self.whole_data.append(np.array([training_data,test_data,feedback_data]))

            self.whole_data = np.array(self.whole_data)

            self.whole_data_len = len(self.whole_data)
            self.output_data_space = np.array([self.encoding_list[range(2)]])

    def state_map(self, neuron_set, alpha_list):
        state_list = list()
        if self.encoding_type == "0to1" :
            for i, alpha in enumerate(alpha_list) :
                state_list.append((neuron_set[i].space_num * alpha).astype(int))

        return state_list

    def state_dict(self, neuron_set, state_list):
        state_dict = dict()
        for i, neuron in enumerate(neuron_set):
            state_dict[neuron.name] = state_list[i]

        return state_dict

    def get_distance(self, neuron, sol, ans):
        return neuron.D[sol][ans]

    def debug_neuron_set(self, neuron_set):
        for i, neuron in enumerate(neuron_set):
            neuron.debug(True)

    def get_debug_info(self, neuron_set):
        debug_info = dict()
        for i, neuron in enumerate(neuron_set):
            debug_info[neuron.name] = neuron.debug_info

        return debug_info

    def get_debug_output_probability(self, neuron_set, output_state_space):
        output_probability_whole = list()
        for i, neuron in enumerate(neuron_set):
            value_history = neuron.debug_info["value_history"]

            output_probability = np.zeros(neuron.space_num)
            for value in value_history :
                output_probability += np.array(neuron.D[value].numpy())

            output_probability = output_probability
            output_probability = output_probability/ sum(output_probability)
            output_probability_whole.append(output_probability)

        return output_probability_whole



    def intput_output_check(self):
        """
        check input, output neuron number of simulator
        """
        input_neuron_num = self.simulator.get_type_neuron_number(0)
        output_neuron_num = self.simulator.get_type_neuron_number(2)
        feedback_neuron_num = self.simulator.get_type_neuron_number(3)

        if input_neuron_num < self.input_num:
            print(f"input neuron number : {input_neuron_num},is less then {self.input_num}")
            return False

        if output_neuron_num < self.output_num:
            print(f"output neuron number : {output_neuron_num},is less then {self.output_num}")
            return False

        if feedback_neuron_num < self.feedback_num:
            print(f"feedback neuron number : {feedback_neuron_num},is less then {self.feedback_num}")
            return False

        return True