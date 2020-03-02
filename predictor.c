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

#define tempPred(p) (p == WN || p == SN) ? NOTTAKEN : TAKEN

// gshare
uint32_t branch_history;
uint8_t *pattern_history_table;

// tournament
uint32_t *localPHT;
uint8_t *localBHT;
uint8_t *globalBHT;
uint8_t *tPredictor;
uint32_t globalhistory;

// custom

#define weight_num 427
#define hist_width 20
#define customIndex(x) ((x * weight_num) % hist_width)

uint8_t custom_thres_train = 0;
int16_t custom_weight[weight_num][hist_width + 1];
int16_t custom_gHistory[hist_width];
int32_t custom_train_theta;
uint8_t custom_prediction = NOTTAKEN;

//------------------------------------//
// gshare
//------------------------------------//

void gshare_init()
{
	size_t PHTSize = (1 << ghistoryBits) * sizeof(uint8_t);

	branch_history = 0;

	pattern_history_table = (uint8_t *)malloc(PHTSize);
	memset(pattern_history_table, 1, PHTSize);
}

uint8_t gshare_predict(uint32_t pc)
{
	uint32_t index = (branch_history ^ pc) & ((1 << ghistoryBits) - 1);
	uint8_t prediction = pattern_history_table[index];
	uint8_t final = tempPred(prediction);
	return final;
}

void gshare_train(uint32_t pc, uint8_t outcome)
{
	uint32_t index = (branch_history ^ pc) & ((1 << ghistoryBits) - 1);
	uint8_t prediction = pattern_history_table[index];

	if (outcome == TAKEN && prediction != ST)
	{
		pattern_history_table[index]++;
	}
	if (outcome == NOTTAKEN && prediction != SN)
	{
		pattern_history_table[index]--;
	}

	branch_history = branch_history << 1;
	branch_history = branch_history | outcome;
}

//------------------------------------//
// tournament
//------------------------------------//

void tournament_init()
{
	globalhistory = 0;

	localPHT = malloc((1 << pcIndexBits) * sizeof(uint32_t));
	localBHT = malloc((1 << lhistoryBits) * sizeof(uint8_t));
	globalBHT = malloc((1 << ghistoryBits) * sizeof(uint8_t));
	tPredictor = malloc((1 << ghistoryBits) * sizeof(uint8_t));

	memset(localPHT, SN, (1 << pcIndexBits) * sizeof(uint32_t));
	memset(localBHT, WN, (1 << lhistoryBits) * sizeof(uint8_t));
	memset(globalBHT, WN, (1 << ghistoryBits) * sizeof(uint8_t));
	memset(tPredictor, WN, (1 << ghistoryBits) * sizeof(uint8_t));
}

uint8_t tournament_predict_local(uint32_t pc)
{
	uint32_t pcIndex = pc & ((1 << pcIndexBits) - 1);
	uint32_t index = localPHT[pcIndex];
	uint8_t prediction = localBHT[index];

	prediction = tempPred(prediction);

	return prediction;
}
uint8_t tournament_predict_global(uint32_t pc)
{
	uint32_t index = (globalhistory & ((1 << ghistoryBits) - 1));
	uint8_t prediction = globalBHT[index];

	prediction = tempPred(prediction);
	
	return prediction;
}

uint8_t tournament_predict(uint32_t pc)
{
	uint32_t index = globalhistory & ((1 << ghistoryBits) - 1);
	uint32_t predictor = tPredictor[index];
	
	if (predictor == SN || predictor == WN)
	{
		return tournament_predict_local(pc);
	}
	if (predictor == ST || predictor == WT)
	{
		return tournament_predict_global(pc);
	}
}

void tournament_train(uint32_t pc, uint8_t outcome)
{
	uint8_t local = tournament_predict_local(pc);
	uint8_t global = tournament_predict_global(pc);

	if (local != global)
	{
		uint8_t status = -1;
		uint8_t prediction = tPredictor[globalhistory];

		if (global == outcome)
		{
			status = NOTTAKEN;
		}

		if (local == outcome)
		{
			status = TAKEN;
		}

		if (status == TAKEN && prediction != ST)
		{
			tPredictor[globalhistory]++;
		}
		if (status == NOTTAKEN && prediction != SN)
		{
			tPredictor[globalhistory]--;
		}
	}

	uint32_t pcIndex = pc & ((1 << pcIndexBits) - 1);
	uint32_t BHTIndex = localPHT[pcIndex];
	uint8_t prediction = localBHT[BHTIndex];

	if (outcome == TAKEN && prediction != ST)
	{
		localBHT[BHTIndex]++;
	}
	if (outcome == NOTTAKEN && prediction != SN)
	{
		localBHT[BHTIndex]--;
	}
	
	localPHT[pcIndex] = (BHTIndex << 1) & ((1 << lhistoryBits) - 1) | outcome;

	prediction = globalBHT[BHTIndex];

	if (outcome == TAKEN && prediction != ST)
	{
		globalBHT[globalhistory]++;
	}
	if (outcome == NOTTAKEN && prediction != SN)
	{
		globalBHT[globalhistory]--;
	}
	
	globalhistory = (globalhistory << 1) & ((1 << ghistoryBits) - 1) | outcome;
}

//------------------------------------//
// custom
//------------------------------------//

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

void custom_init()
{
	custom_train_theta = (2 * hist_width + 14);
	memset(custom_weight, 0, sizeof(int16_t) * weight_num * (hist_width + 1));
	memset(custom_gHistory, 0, sizeof(uint16_t) * hist_width);
}

uint8_t custom_predict(uint32_t pc)
{
	uint32_t index = customIndex(pc);
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
	uint32_t index = customIndex(pc);
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
