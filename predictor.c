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
//include the file which you write

//
// TODO:Student Information
//
const char *studentName = "Kunlin YANG";
const char *studentID = "KY PID";
const char *email = "k6yang@eng.ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = {"Static", "Gshare",
						 "Tournament", "Custom"};

int ghistoryBits; // Number of bits used for Global History
int lhistoryBits; // Number of bits used for Local History
int pcIndexBits;  // Number of bits used for PC index
int bpType;		  // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//
uint32_t pcindex_bits = 8;
uint32_t ghistory; //storing the ghistory which is a global variable
uint8_t *gshareBHT;
uint32_t *pc_lhr;		 //storing the local history of each of the branches according to pc, pc -> lhr
uint8_t *local_pred_BHT; //storing the BHT of each of the brances according to the local history, lhr -> prediction
uint8_t *global_pred_BHT;
uint8_t *meta_predictor;
uint8_t global_outcome;
uint8_t local_outcome;
uint32_t outcome_new;
uint8_t index_lsb;
uint32_t globalhistory;

// custom data structure

#define weight_num 427
#define hist_width 20

uint8_t custom_thres_train = 0;
int16_t custom_weight[weight_num][hist_width + 1]; // 1 extra is for the bias
int16_t custom_gHistory[hist_width];
int32_t custom_train_theta;
uint8_t custom_prediction = NOTTAKEN;

//------------------------------------//
//          Helper Functions          //
//------------------------------------//

uint8_t state_prediction(uint8_t prediction)
{
	uint8_t predicted_outcome;
	if (prediction == SN || prediction == WN)
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
void prediction_update(uint8_t *bht_index, uint8_t outcome)
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

uint32_t get_index_custom(uint32_t x)
{
	return ((x * hist_width) % weight_num);
}

void prediction_update_custom(int16_t *weight, uint8_t outcome)
{
	if (outcome == TAKEN)
	{
		if (*weight != 127)
		{
			(*weight)++;
		}
	}
	else if (outcome == NOTTAKEN)
	{
		if (*weight != -126)
		{
			(*weight)--;
		}
	}
}

//------------------------------------//
// gshare
//------------------------------------//

void gshare_init()
{
	ghistory = 0;
	gshareBHT = malloc((1 << ghistoryBits) * sizeof(uint8_t));   //this is like an array pointer in which the prediction is getting stored, suppose 10 bits for index, 2^10 prediction bits
	memset(gshareBHT, 1, (1 << ghistoryBits) * sizeof(uint8_t)); //predict weakly taken for all the indices currently, so each prediction is taken as 1 byte
}

uint8_t
gshare_predict(uint32_t pc)
{
	ghistory = ghistory & ((1 << ghistoryBits) - 1);
	pc = (pc & ((1 << ghistoryBits) - 1)); //shifting the pc by two bits and getting index of 'ghistory' bits
	uint32_t index = (ghistory ^ pc);	  //FIXME: assuming ghistory is the no of bits we are dealing with
	uint8_t prediction = gshareBHT[index];
	global_outcome = state_prediction(prediction);
	return global_outcome;
}

//The function is used to train the bht entries with respect to the xor between the present GHR and the incoming PC
void gshare_train(uint32_t pc, uint8_t outcome)
{
	pc = (pc & ((1 << ghistoryBits) - 1));
	//printf("%d", gshareBHT[pc ^ ghistory] );
	ghistory = ghistory & ((1 << ghistoryBits) - 1);
	prediction_update(&(gshareBHT[pc ^ ghistory]), outcome);
	ghistory = ghistory << 1;
	ghistory = ghistory & ((1 << ghistoryBits) - 1);
	ghistory = ghistory | outcome;
}

//------------------------------------//
// tournament
//------------------------------------//

void tournament_init()
{
	//local prediction tables
	//printf("pcindex_bits %d", pcindex_bits);
	//printf("lhistoryBits %d", lhistoryBits);
	//printf("ghistoryBits %d", ghistoryBits);
	pc_lhr = malloc((1 << pcIndexBits) * sizeof(uint32_t));
	local_pred_BHT = malloc((1 << lhistoryBits) * sizeof(uint8_t));
	memset(pc_lhr, 0, (1 << pcIndexBits) * sizeof(uint32_t));		   //setting the local history table
	memset(local_pred_BHT, WN, (1 << lhistoryBits) * sizeof(uint8_t)); //setting the prediction corresponding to the history table

	//global predictor tables
	globalhistory = 0;
	global_pred_BHT = malloc((1 << ghistoryBits) * sizeof(uint8_t));   //this is like an array pointer in which the prediction is getting stored, suppose 10 bits for index, 2^10 prediction bits
	memset(global_pred_BHT, 1, (1 << ghistoryBits) * sizeof(uint8_t)); //predict weakly taken for all the indices currently, so each prediction is taken as 1 byte

	//meta predictor tables
	meta_predictor = malloc((1 << ghistoryBits) * sizeof(uint8_t));
	memset(meta_predictor, WN, (1 << ghistoryBits) * sizeof(uint8_t));
}

uint8_t
tournament_predict_local(uint32_t pc)
{
	pc = (pc & ((1 << pcIndexBits) - 1));
	uint32_t index = pc_lhr[pc];
	uint8_t prediction = local_pred_BHT[index]; //the index is the local history register for that particular pc
	local_outcome = state_prediction(prediction);
	return local_outcome;
}
uint8_t
tournament_predict_global(uint32_t pc)
{
	uint32_t index = globalhistory & ((1 << ghistoryBits) - 1);
	uint8_t prediction = global_pred_BHT[index];
	global_outcome = state_prediction(prediction);
	return global_outcome;
}

uint8_t
tournament_predict(uint32_t pc)
{
	uint32_t index = globalhistory & ((1 << ghistoryBits) - 1);
	uint32_t predictor_choice = meta_predictor[index]; //the index of the meta predictor is given by the global history bits
	tournament_predict_local(pc);
	tournament_predict_global(pc);
	if (predictor_choice == SN || predictor_choice == WN)
	{ //while training, correctness of the global prediction decrements the metapredictor
		return global_outcome;
	}
	else if (predictor_choice == ST || predictor_choice == WT)
	{ //while training correctness of the local predictor increments the metapredictor
		return local_outcome;
	}
}

void tournament_train(uint32_t pc, uint8_t outcome)
{
	//updating the meta predictor only in the case localpredictor != globalpredictor
	if (local_outcome != global_outcome)
	{
		if (local_outcome == outcome)
		{
			prediction_update(&meta_predictor[globalhistory], TAKEN);
		}
		else if (global_outcome == outcome)
		{
			prediction_update(&meta_predictor[globalhistory], NOTTAKEN);
		}
	}

	//updating the local_pred_BHT and pc_lhr
	uint32_t pc_index = pc & ((1 << pcIndexBits) - 1);
	uint32_t lhr_index = pc_lhr[pc_index];
	prediction_update(&local_pred_BHT[lhr_index], outcome);
	pc_lhr[pc_index] = pc_lhr[pc_index] << 1;
	pc_lhr[pc_index] = pc_lhr[pc_index] & ((1 << lhistoryBits) - 1);
	pc_lhr[pc_index] = pc_lhr[pc_index] | outcome;

	//updating the global_pred_BHT
	prediction_update(&global_pred_BHT[globalhistory], outcome);
	globalhistory = globalhistory << 1;
	globalhistory = globalhistory & ((1 << ghistoryBits) - 1);
	globalhistory = globalhistory | outcome;
}

//------------------------------------//
// custom
//------------------------------------//

void custom_init()
{
	custom_train_theta = (2 * hist_width + 14);
	memset(custom_weight, 0, sizeof(int16_t) * weight_num * (hist_width + 1));
	memset(custom_gHistory, 0, sizeof(uint16_t) * hist_width);
}

uint8_t custom_predict(uint32_t pc)
{
	uint32_t index = get_index_custom(pc);
	int16_t pred_out = custom_weight[index][0];

	for (int i = 1; i <= hist_width; i++)
	{
		pred_out = pred_out + (custom_gHistory[i - 1] ? custom_weight[index][i] : -custom_weight[index][i]);
	}

	custom_prediction = (pred_out >= 0) ? TAKEN : NOTTAKEN;
	custom_thres_train = (pred_out < custom_train_theta && pred_out > -custom_train_theta) ? 1 : 0;

	return custom_prediction;
}

void custom_train(uint32_t pc, uint8_t outcome)
{
	uint32_t index = get_index_custom(pc);
	if ((custom_prediction != outcome) || custom_thres_train)
	{
		prediction_update_custom(&(custom_weight[index][0]), outcome);
		for (int i = 1; i <= hist_width; i++)
		{
			uint8_t predict = custom_gHistory[i - 1];
			if (outcome == predict)
				prediction_update_custom(&(custom_weight[index][i]), 1);
			else
				prediction_update_custom(&(custom_weight[index][i]), 0);
		}
	}

	for (int i = hist_width - 1; i > 0; i--)
	{
		custom_gHistory[i] = custom_gHistory[i - 1];
	}
	custom_gHistory[0] = outcome;
}

//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//
void init_predictor()
{
	//
	//TODO: Initialize Branch Predictor Data Structures
	switch (bpType)
	{
	case STATIC:
		return;
	case GSHARE:
		gshare_init();
		break;
	case TOURNAMENT:
		tournament_init();
		break;
	case CUSTOM:
		custom_init();
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
	switch (bpType)
	{
	case STATIC:
		return TAKEN;
	case GSHARE:
		return gshare_predict(pc);
	case TOURNAMENT:
		return tournament_predict(pc);
	case CUSTOM:
		return custom_predict(pc);
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
void train_predictor(uint32_t pc, uint8_t outcome)
{
	switch (bpType)
	{
	case STATIC:
		break;
	case GSHARE:
		gshare_train(pc, outcome);
		break;
	case TOURNAMENT:
		tournament_train(pc, outcome);
		break;
	case CUSTOM:
		custom_train(pc, outcome);
		break;
	}
}
