#include <iostream>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <ctime>
#include <regex>
#include <random>
#include "Node.h"

void connectNetwork();
void printNetwork();
void initWeights();
void runANN(std::array<int, 5>);
void backPass();
void adjustWeights();
void resetValues();

//global variables
struct glob{
	std::vector<Node> input; // holds the input nodes
	std::vector<std::vector<Node>> hidden; //hold the hidden layer nodes
	std::vector<Node> output; //hold the output nodes
	int numEpochs; //defines maximum number of epochs
	std::vector<std::array<int,5>> trainingSet; // first 4 actual bits, 5th is the parity bit
	int numSetsTU; //sets how many training sets to use
	double learnRate; // sets the learning rate

} global;

//initializes network: takes in the architecture of the network
void initANN(int input, int hidden[], int numHiddenLayers, int output){
	//asks for user to enter learning rate
	printf("Please enter a learning rate (as decimal): ");
	std::cin >> global.learnRate;

	//creates the appropriate amount of nodes in each layer
	printf("Input Size Specified: %d\n", input);
	for (int i = 0; i < input; i++){
		Node n(global.learnRate);
		n.name = "In: " + std::to_string(i);
		global.input.push_back(n);
	}

	//supports multiple hidden layers
	printf("Number of hidden Layers: %d\n", numHiddenLayers);
	for (int layer = 0; layer < numHiddenLayers; layer++){
		std::vector<Node> h;
		global.hidden.push_back(h);
		for (int i = 0; i < hidden[layer]; i++){
			Node n(global.learnRate);
			n.name = "h" + std::to_string(layer) + "-" + std::to_string(i);
			global.hidden.at(layer).push_back(n);
		}
	}

	for (int i = 0; i < output; i++){
		Node n(global.learnRate);
		n.name = "Out: " + std::to_string(i);
		global.output.push_back(n);
	}

	//connects the network and then sets up the weights
	connectNetwork();
	initWeights();
}

//connects the network
void connectNetwork(){
	//sets the inputs of the hidden nodes to each of the input nodes
	//and sets the output of the input nodes as the hidden nodes
	for (int i = 0; i < global.input.size(); i++){
		for (int j = 0; j < global.hidden.at(0).size(); j++){
			global.input.at(i).addOutput(&global.hidden.at(0).at(j));
			global.hidden.at(0).at(j).addInput(&global.input.at(i));
		}
	}
	//connects hidden to hidden and hidden to output the same way as above
	//does not support no hidden layers
	for (int layer = 0; layer < global.hidden.size(); layer++){
		if (layer + 1 < global.hidden.size()){
			for (int nLayer = layer+1; nLayer < global.hidden.size(); nLayer++){ // TODO this should be a variable
				for (int i = 0; i < global.hidden.at(layer).size(); i++){
					for (int j = 0; j < global.hidden.at(nLayer).size(); j++){
						global.hidden.at(layer).at(i).addOutput(&global.hidden.at(nLayer).at(j));
						global.hidden.at(nLayer).at(j).addInput(&global.hidden.at(layer).at(i));
					}
				}
			}
		}else{
			for (int i = 0; i < global.hidden.at(layer).size(); i++){
				for (int j = 0; j < global.output.size(); j++){
					global.hidden.at(layer).at(i).addOutput(&global.output.at(j));
					global.output.at(j).addInput(&global.hidden.at(layer).at(i));
				}
			}
		}
	}
}

//initializes the weights by calling each nodes to generate the amount of weights needed
void initWeights(){
	for (int i = 0; i < global.input.size(); i++){
		global.input.at(i).initWeights();
	}
	for (int i = 0; i < global.hidden.size(); i++){
		for (int j = 0; j < global.hidden.at(i).size(); j++){
			global.hidden.at(i).at(j).initWeights();
		}
	}
	for (int i = 0; i < global.output.size(); i++){
		global.output.at(i).initWeights();
	}

}


//does the forward pass through the network on a given input
void runANN(std::array<int, 5> values){			
	for (int i = 0; i < global.input.size(); i++){
		global.input.at(i).setValue(values.at(i));
		//initial push doesn't pass the node value through the sigmoid function
		global.input.at(i).initialPush(); 
	}
	for (int i = 0; i < global.hidden.size(); i++){
		for (int j = 0; j < global.hidden.at(i).size(); j++){
			//forward does use the sigmoid function
			global.hidden.at(i).at(j).forward();
		}
	}		
	for (int i = 0; i < global.output.size(); i++){
		global.output.at(i).forward();
	}
}

//loops through the training examples, and does the correction logic
void trainANN(){

	std::clock_t start = std::clock(); // get timer to check how long it takes to train
	//loop through the number of epochs specified
	for (int epoch = 0; epoch < global.numEpochs; epoch++){
		if ((epoch + 1) % 500 == 0){ //only print out every 500 epochs for speed
			printf("Epoch: %d/%d\n", epoch + 1, global.numEpochs);
			//printNetwork();
		}
		//randomly shuffle the training examples
		auto engine = std::default_random_engine{};
		std::shuffle(global.trainingSet.begin(), global.trainingSet.end(), engine);
		
		//set the correct number predicted to 100% (all of the examples that it will use)
		int correctNum = global.numSetsTU;

		//loop through each training example once
		for (int train = 0; train < global.numSetsTU; train++){	

			//do the forward pass
			runANN(global.trainingSet.at(train));
			

			bool incorrect = false; // assume the correct result was predicted
			//loop through all output nodes
			for (int i = 0; i < global.output.size(); i++){
				double rawRes = global.output.at(i).getOutput();//get raw output
				int result = rawRes < 0.5 ? 0 : 1; // round the result to get prediction
				int correctRes = global.trainingSet.at(train).at(4); // get correct result
	//			if (correctRes != result){ //if they are not equal
					incorrect = true; //then the prediction was incorrect
					double err = correctRes - rawRes; //calculate the error
					global.output.at(i).addError(err); //add the error to the output node
	//			}
			}
			if (incorrect){ //if the result was incorrect then
				correctNum--;//decrease correct num
				backPass();//and do the back propogation
			}
			resetValues(); //reset the value, output and error at all nodes to reset network
		}
		//printf("Correct: %d/%d\n", correctNum, global.numSetsTU); //used to print network for debugging
		if (correctNum >= global.numSetsTU * 1.00){ // if all predicted then break
			printf("All test cases predicted!\n");
			break;
		}

	}
	//print out how long it took to train
	printf("Time to train: %f\n", ((std::clock() - start) / (double)(CLOCKS_PER_SEC / 1000)));
}


void useANN(){
	//ask print out stats
	printf("\nLearning rate: %f\n", global.learnRate);
	printf("Network Type: %d-%d-%d\n\n", global.input.size(), global.hidden.at(0).size(), global.output.size());
	printf("Results for inputs:\n");
	//go through and print out all training examples and info
	for (std::array<int, 5> inp : global.trainingSet){
		runANN(inp);
		for (int i = 0; i < global.trainingSet.at(0).size()-1; i++){
			printf("%d",inp.at(i));
		}
		printf("\t");
		for (int i = 0; i < global.output.size(); i++){
			double rawRes = global.output.at(i).getOutput();
			int result = rawRes < 0.5 ? 0 : 1;
			
			printf("Expected: %d   Result: %d   %f%% accuracy   ",inp.at(4), result, result == 0 ? (((0.5 - rawRes) / 0.5) * 100) : (((rawRes - 0.5) / 0.5) * 100));
			printf("Raw result: %f\n", rawRes);
			resetValues();
		}
	}

	//allow user to experiment with inputs
	while (true){
		printf("Enter q at any time to quit!\nEnter the 4 bits\n");
		printf("\n\n");
		std::string input;
		std::cin >> input;
		
		//check if the user wants to quit
		if (input.find('q') != std::string::npos || input.find('Q') != std::string::npos)
			exit(0);
		//check if input is wrong size
		if (input.size() != 4){
			printf("Please enter the correct number of bits!\n");
		}

		//convert string to char array
		char inputArr[5];
		strcpy_s(inputArr, input.c_str());
		
		
		std::cmatch cm;
		std::regex reg("([^01]+)");
		//compare regex to input to make sure only 1's and 0's were entered
		if (std::regex_search(inputArr, cm, reg)){
			printf("Please enter valid bits!\n");
		}else
		{
			//if okay, then tag extra character to end so algorithm will work
			inputArr[4] = '0';
			input += "0";

			//convert to int array
			std::array<int, 5> inp;
			for (int i = 0; i < sizeof(inputArr); i++){
				inp[i] = ((int)inputArr[i]) - 48;
			}

			//run the forward pass
			runANN(inp);

			//gather and display results
			for (int i = 0; i < global.output.size(); i++){
				double rawRes = global.output.at(i).getOutput();
				int result = rawRes < 0.5 ? 0 : 1;
				printf("Raw result: %f\n", rawRes);
				printf("Result is %d with %f%% accuracy\n", result, result == 0 ? (((0.5 - rawRes) / 0.5) * 100) : (((rawRes - 0.5) / 0.5) * 100));
				resetValues(); //reset the network
			}
		}
	
	}

}


//does the backpass, calculates error at each node
void backPass(){
	for (int i = 0;i < global.output.size(); i++){
		global.output.at(i).findError();
	}

	for (int i = global.hidden.size() - 1; i >= 0; i--){
		for (int j = 0; j < global.hidden.at(i).size(); j++){
			global.hidden.at(i).at(j).findError();
		}
	}
	adjustWeights();

}

//adjusts the connection weights
void adjustWeights(){

	for (int i = 0; i < global.hidden.size(); i++){
		for (int j = 0; j < global.hidden.at(i).size(); j++){
			global.hidden.at(i).at(j).updateWeights();
		}
	}
	
	for (int i = 0; i < global.output.size(); i++){
		global.output.at(i).updateWeights();
	}

}

//resets the network to use again
void resetValues(){
	for (int i = 0; i < global.input.size(); i++){
		global.input.at(i).resetValues();
	}

	for (int i = 0; i < global.hidden.size(); i++){
		for (int j = 0; j < global.hidden.at(i).size(); j++){
			global.hidden.at(i).at(j).resetValues();
		}
	}

	for (int i = 0; i < global.output.size(); i++){
		global.output.at(i).resetValues();
	}

}

//prints the network in a readable form
void printNetwork(){
	printf("Input Layer Size: %d\n", global.input.size());
	for (Node n : global.input){
		n.printConnections();
		printf("\n");
	}

	printf("Hidden Layer Number: %d\n", global.hidden.size());
	
	for (std::vector<Node> v : global.hidden){
		printf("Hidden Layer number of nodes %d: \n", v.size());
		for (Node n : v){
			n.printConnections();
			printf("\n");
		}
	}

	printf("Output Layer size: %d ", global.output.size());
	for (Node n : global.output){
		n.printConnections();
		printf("\n");
	}
}

//reads in the training examples from a file
void readFile(){
	std::ifstream in;
	in.open("../Input/input.txt");

	std::string line;
	//loops while there are things in the file
	while (getline(in, line)){
		if (line.length() == 5){ //all inputs must be 5. 4 for the inputs and 1 for the teacher parity bit
			std::array<int, 5> res;
			for (int i = 0; i < line.length(); i++){
				res[i] = line.at(i) - '0'; //convert to int
			}
			global.trainingSet.push_back(res); //add to collection
		}
	}

	in.close(); //close the file
}


int main(int argc, char* argv[]){
	srand(time(NULL)); //set a random seed
	readFile(); //read in the training examples
	global.numEpochs = 50000; //specify max number of epochs
	global.numSetsTU = 16; //set the number of training sets to use
	global.learnRate = 0.15; //set the learning rate
	//specifies the number of hidden layers and nodes. 15 is the lowest I managed to get to work with a 0.15 learning rate
	int hidden[] = { 24 }; //set the number of hidden nodes in each hidden layer
	initANN(4, hidden, std::end(hidden) - std::begin(hidden), 1); //init the neural net
	trainANN(); //train the neural net
	useANN(); //allow user to use net
}