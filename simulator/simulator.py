import numpy as np
from modules import WPLRNeuron, WPLRSynapse
from collections import OrderedDict
from Args import Args
args = Args()
import matplotlib.pyplot as plt
import networkx as nx

class Simulator() :
    np.random.seed(None)

    # sampling -> input -> training

    # sampling <-> sampling + input : predict <-> real
    # loss entropy : low resource

    def __init__(self):
        self.neurons = OrderedDict()
        self.synapses = OrderedDict()


    def add_neuron(self, neuron_name, neuron_type=1, space_num=100):
        """
        type = 0 : input neuron
        type = 1 : hidden neuron
        tupe = 2 : output neuron
        type = 3 : feedback neuron
        """
        self.neurons[neuron_name] = WPLRNeuron(neuron_name, neuron_type, space_num=space_num)

    def add_synapse(self, synapse_name, pre_neuron_name, post_neuron_name, condition=True):
        pre_neuron = self.neurons[pre_neuron_name]
        post_neuron = self.neurons[post_neuron_name]
        self.synapses[synapse_name] = WPLRSynapse(pre_neuron, post_neuron, condition=condition)

    def init_node(self):
        for neuron_name, neuron in self.neurons.items():
            neuron.init_node()


    def visual_synapse(self, synapse_name, angle_x=90, angle_y=0):
        self.synapses[synapse_name].visual_synapse(angle_x, angle_y)

    def sampling(self):
        for neuron_name, neuron in self.neurons.items():
            neuron.init_probability()

        for synapse_name, synapse in self.synapses.items():
            synapse.propagate_probability()

        for neuron_name, neuron in self.neurons.items():
            neuron.sampling()


    def training_save(self):
        for synapse_name, synapse in self.synapses.items():
            synapse.training_save()

    def training_by_value(self, value=1):
        for synapse_name, synapse in self.synapses.items():
            synapse.training_by_value(value)

    def input(self, input_dict):
        for neuron_name, value in input_dict.items() :
            self.neurons[neuron_name].input(value)


    def get_type_neuron_number(self, type):
        cnt=0
        for neuron_name, neuron in self.neurons.items() :
            if neuron.type == type : cnt += 1

        return cnt

    def get_type_neurons(self, type, neuron_num):
        neuron_list = list()
        neuron_cnt = 0
        for neuron_name, neuron in self.neurons.items():
            if neuron_num <= neuron_cnt :
                break

            if neuron.type == type: neuron_list.append(neuron)

        return neuron_list

    def all_neuron_sleep(self):
        for neuron_name, neuron in self.neurons.items() :
            neuron.sleep()

    def visual_graph(self):
        G = nx.DiGraph()
        node_color = list()
        node_edge_colors = list()
        for neuron_name, neuron in self.neurons.items():
            G.add_node(neuron_name)
            if neuron.sleep == True : node_color.append("tab:blue")
            else : node_color.append("tab:cyan")

            if neuron.type == 0 : node_edge_colors.append("b")
            elif neuron.type == 2: node_edge_colors.append("tab:red")
            elif neuron.type == 3: node_edge_colors.append("m")
            else : node_edge_colors.append("w")

        edge_color = list()
        edge_activation_label = OrderedDict()
        edge_inhibition_label = OrderedDict()
        edge_self_activation_label= OrderedDict()
        edge_self_inhibition_label = OrderedDict()
        for synapse_name, synapse in self.synapses.items():
            G.add_edge(synapse.pre_neuron.name, synapse.post_neuron.name)
            if synapse.sleep == True : edge_color.append("k")
            else : edge_color.append("tab:cyan")
            if synapse.condition == 1 :
                if synapse.pre_neuron.name == synapse.post_neuron.name : edge_self_activation_label[(synapse.pre_neuron.name, synapse.post_neuron.name)] = synapse_name
                else: edge_activation_label[(synapse.pre_neuron.name, synapse.post_neuron.name)] = synapse_name
            else :
                if synapse.pre_neuron.name == synapse.post_neuron.name : edge_self_inhibition_label[(synapse.pre_neuron.name, synapse.post_neuron.name)] = synapse_name
                else: edge_inhibition_label[(synapse.pre_neuron.name, synapse.post_neuron.name)] = synapse_name


        try:
            nx.draw(G, pos=self.pos,
                    with_labels=True,
                    node_color=node_color,
                    edgecolors=node_edge_colors,
                    linewidths=2,
                    node_size=1000,
                    edge_color = edge_color,)
        except :
            self.pos = nx.circular_layout(G)
            nx.draw(G, pos=self.pos,
                    with_labels=True,
                    node_color=node_color,
                    edgecolors=node_edge_colors,
                    linewidths=2,
                    node_size=1000,
                    edge_color=edge_color,)

        rebuild_pos = dict()
        for n, p in self.pos.items():
            rebuild_pos[n] = [p[0], p[1]+0.2 ]

        nx.draw_networkx_edge_labels( G, pos= self.pos, edge_labels=edge_activation_label,font_color="blue",label_pos=0.5)
        nx.draw_networkx_edge_labels( G, pos= self.pos, edge_labels=edge_inhibition_label,font_color="red",label_pos=0.5)
        nx.draw_networkx_edge_labels(G, pos=rebuild_pos, edge_labels=edge_self_activation_label, font_color="blue",label_pos=0.5)
        nx.draw_networkx_edge_labels(G, pos=rebuild_pos, edge_labels=edge_self_inhibition_label, font_color="red",label_pos=0.5)

        plt.show()
        return


