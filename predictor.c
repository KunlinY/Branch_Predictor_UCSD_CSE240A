//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include "predictor.h"
#include <stdlib.h>
#include "perceptron.h"
//include the file which you write

//
// TODO:Student Information
//
const char *studentName = "NAME";
const char *studentID   = "PID";
const char *email       = "EMAIL";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = { "Static", "Gshare",
                          "Tournament", "Custom" };

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;       // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//
uint32_t pcindex_bits = 8;
uint32_t ghistory ;  //storing the ghistory which is a global variable
uint8_t *gshareBHT;
uint32_t *pc_lhr;       //storing the local history of each of the branches according to pc, pc -> lhr
uint8_t *local_pred_BHT; 		//storing the BHT of each of the brances according to the local history, lhr -> prediction
uint8_t *global_pred_BHT;
uint8_t *meta_predictor;
uint8_t global_outcome;
uint8_t local_outcome;
uint32_t outcome_new;
uint8_t index_lsb;
uint32_t globalhistory;



//
//TODO: Add your own Branch Predictor data structures here
//


//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
uint8_t state_prediction(uint8_t prediction)
{
	uint8_t predicted_outcome;
	if(prediction == SN || prediction == WN)
	{
		predicted_outcome = NOTTAKEN;
	}
	else 
	{
		predicted_outcome = TAKEN;
	}
	return predicted_outcome;
}
//The function is used to update the bht entry for the branch wrt to the present history bits
void
prediction_update(uint8_t *bht_index, uint8_t outcome)
{
	if (outcome == TAKEN)
	{
		if (*bht_index != 3) //strongly taken state
		{
			(*bht_index)++;   
		}
	}
	else if (outcome == NOTTAKEN)
	{
		if (*bht_index != 0) //strongly not taken state
		{
			(*bht_index)--;
		}
	}
}

void
tournament_predictor_init()
{
	//local prediction tables
	//printf("pcindex_bits %d", pcindex_bits);
	//printf("lhistoryBits %d", lhistoryBits);
	//printf("ghistoryBits %d", ghistoryBits);
	pc_lhr = malloc((1 << pcIndexBits) * sizeof(uint32_t));
	local_pred_BHT = malloc((1 << lhistoryBits) * sizeof(uint8_t));
	memset(pc_lhr, 0, (1 << pcIndexBits) * sizeof(uint32_t)); //setting the local history table
	memset(local_pred_BHT, WN, (1 << lhistoryBits) * sizeof(uint8_t));//setting the prediction corresponding to the history table
	
	
	//global predictor tables
	globalhistory = 0;
	global_pred_BHT = malloc((1 << ghistoryBits) * sizeof(uint8_t)); //this is like an array pointer in which the prediction is getting stored, suppose 10 bits for index, 2^10 prediction bits
	memset(global_pred_BHT, 1, (1 << ghistoryBits) * sizeof(uint8_t));	//predict weakly taken for all the indices currently, so each prediction is taken as 1 byte 
	
	//meta predictor tables
	meta_predictor = malloc((1 << ghistoryBits) * sizeof(uint8_t));
	memset(meta_predictor, WN, (1 << ghistoryBits) * sizeof(uint8_t));
}
void 
custom_predictor_init()
{
	//local prediction tables
	//printf("pcindex_bits %d", pcindex_bits);
	//printf("lhistoryBits %d", lhistoryBits);
	//printf("ghistoryBits %d", ghistoryBits);
	pcIndexBits = 10;
	lhistoryBits = 10;
	ghistoryBits = 10;
	tournament_predictor_init();
}

uint8_t
make_gshare_prediction(uint32_t pc)
{
	ghistory = ghistory & ((1 << ghistoryBits) - 1);
	pc = (pc  & ((1 << ghistoryBits) - 1));   	//shifting the pc by two bits and getting index of 'ghistory' bits
	uint32_t index = (ghistory ^ pc);   //FIXME: assuming ghistory is the no of bits we are dealing with
	uint8_t prediction = gshareBHT[index];
	global_outcome = state_prediction(prediction);
	return global_outcome;
}
uint8_t
make_tournament_local_prediction(uint32_t pc)
{
	pc = (pc & ((1 << pcIndexBits) - 1)); 
	uint32_t index = pc_lhr[pc];
	uint8_t prediction = local_pred_BHT[index]; //the index is the local history register for that particular pc
	local_outcome = state_prediction(prediction);
	return local_outcome;
}
uint8_t
make_tournament_global_prediction(uint32_t pc)
{
	uint32_t index = globalhistory & ((1 << ghistoryBits) - 1);
	uint8_t prediction = global_pred_BHT[index];
	global_outcome = state_prediction(prediction);
	return global_outcome;
}
uint8_t
make_tournament_prediction(uint32_t pc)
{
	uint32_t index = globalhistory & ((1 << ghistoryBits) - 1);
	uint32_t predictor_choice = meta_predictor[index];  //the index of the meta predictor is given by the global history bits
	make_tournament_local_prediction(pc);
	make_tournament_global_prediction(pc);
	if (predictor_choice == SN || predictor_choice == WN){ //while training, correctness of the global prediction decrements the metapredictor
		return global_outcome;
	}
	else if (predictor_choice == ST || predictor_choice == WT){ //while training correctness of the local predictor increments the metapredictor
		return local_outcome;
	}
}
uint8_t
make_custom_prediction(uint32_t pc)
{
	make_tournament_prediction(pc);
}

//The function is used to train the bht entries with respect to the xor between the present GHR and the incoming PC
void
train_gshare_predictor(uint32_t pc, uint8_t outcome)
{
	pc = (pc & ((1 << ghistoryBits) - 1));
	//printf("%d", gshareBHT[pc ^ ghistory] );
	ghistory = ghistory & ((1 << ghistoryBits) - 1);
	prediction_update(&(gshareBHT[pc ^ ghistory]), outcome);
	ghistory = ghistory <<	1;
	ghistory = ghistory & ((1 << ghistoryBits) - 1);
	ghistory = ghistory | outcome;
}

void
train_tournament_predictor(uint32_t pc, uint8_t outcome)
{
	//updating the meta predictor only in the case localpredictor != globalpredictor
	if (local_outcome != global_outcome){
		if (local_outcome == outcome){
			prediction_update(&meta_predictor[globalhistory], TAKEN);
		}
		else if (global_outcome == outcome){
			prediction_update(&meta_predictor[globalhistory], NOTTAKEN);
		}
	}
	
	//updating the local_pred_BHT and pc_lhr
	uint32_t pc_index = pc & ((1 << pcIndexBits) - 1);
	uint32_t lhr_index = pc_lhr[pc_index];
	prediction_update(&local_pred_BHT[lhr_index], outcome);
	pc_lhr[pc_index] = pc_lhr[pc_index] << 1;
	pc_lhr[pc_index] = pc_lhr[pc_index] & ((1 << lhistoryBits) - 1);
	pc_lhr[pc_index] = pc_lhr[pc_index]|outcome;
	
	//updating the global_pred_BHT
	prediction_update(&global_pred_BHT[globalhistory], outcome);
	globalhistory = globalhistory << 1;
	globalhistory = globalhistory & ((1 << ghistoryBits) - 1);
	globalhistory = globalhistory | outcome;
}
void
train_custom_predictor(uint32_t pc, uint8_t outcome)
{
	train_tournament_predictor(pc,outcome);
}


void
init_predictor()
{
  //
  //TODO: Initialize Branch Predictor Data Structures
  switch(bpType){
	case STATIC: 
		return;
	case GSHARE:
		ghistory = 0;
		gshareBHT = malloc((1 << ghistoryBits) * sizeof(uint8_t)); //this is like an array pointer in which the prediction is getting stored, suppose 10 bits for index, 2^10 prediction bits
		memset(gshareBHT, 1, (1 << ghistoryBits) * sizeof(uint8_t));	//predict weakly taken for all the indices currently, so each prediction is taken as 1 byte
		break;
	case TOURNAMENT:
		tournament_predictor_init();
		break;
	case CUSTOM:
		perceptron_predictor_init();
		//custom_predictor_init();
		break;
		
		
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint8_t
make_prediction(uint32_t pc)
{

  // Make a prediction based on the bpType
  switch (bpType) {
    case STATIC:
      return TAKEN;
    case GSHARE:
		return make_gshare_prediction(pc);
    case TOURNAMENT:
		return make_tournament_prediction(pc);
    case CUSTOM:
		return make_perceptron_prediction(pc);
		//return make_custom_prediction(pc);
		//return perceptron_predict(pc);
    default:
      break;
  }
  

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//
void
train_predictor(uint32_t pc, uint8_t outcome)
{
  //Train the BHT and the indices according to the history
  switch (bpType) {
	case STATIC:
	  break;
	case GSHARE:
		train_gshare_predictor(pc, outcome); //forming the table with the address and the prediction
		break;
	case TOURNAMENT:
		train_tournament_predictor(pc, outcome);
		break;
	case CUSTOM:
		//train_custom_predictor(pc, outcome);
		//train_perceptron_predictor(pc, outcome);
		perceptron_train(pc, outcome);
		break;
  }
}
