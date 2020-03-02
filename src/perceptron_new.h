#include <stdio.h>
#include <string.h>

#define weight_num 427
#define hist_width 20


uint8_t perceptron_thres_train = 0;
int16_t perceptron_weight[weight_num][hist_width + 1];// 1 extra is for the bias
int16_t perceptron_gHistory[hist_width];
int32_t perceptron_train_theta;
uint8_t perceptron_prediction = NOTTAKEN;

uint32_t get_index(uint32_t x)
{
	return ((x * hist_width) % weight_num);
}
void perceptron_update(int16_t* weight, uint8_t outcome){
  if(outcome == 1){
    if(*weight != 127){
	  (*weight)++;
    }
  }else{
    if(*weight != -126){
      (*weight)--;
    }
  }
}

void perceptron_predictor_init(){
  perceptron_train_theta = (2 * hist_width + 14);
  memset(perceptron_weight, 0, sizeof(int16_t) * weight_num * (hist_width + 1));
  memset(perceptron_gHistory, 0, sizeof(uint16_t) * hist_width);
}



uint8_t make_perceptron_prediction(uint32_t pc){
  uint32_t index = get_index(pc);
  int16_t pred_out = perceptron_weight[index][0];

  for(int i = 1 ; i <= hist_width; i++){
    pred_out = pred_out + (perceptron_gHistory[i-1] ? perceptron_weight[index][i] : -perceptron_weight[index][i]);
  }

  perceptron_prediction = (pred_out >= 0) ? TAKEN : NOTTAKEN;
  perceptron_thres_train = (pred_out < perceptron_train_theta && pred_out > -perceptron_train_theta) ? 1 : 0;

  return perceptron_prediction;
}

void perceptron_train(uint32_t pc, uint8_t outcome){
  uint32_t index = get_index(pc);
  if((perceptron_prediction != outcome) || perceptron_thres_train){
    perceptron_update(&(perceptron_weight[index][0]), outcome);
    for(int i = 1 ; i <= hist_width ; i++){
      uint8_t predict = perceptron_gHistory[i-1];
	  if(outcome == predict)
		perceptron_update(&(perceptron_weight[index][i]), 1);
	  else
		perceptron_update(&(perceptron_weight[index][i]), 0);
    }

  }

  for(int i = hist_width - 1; i > 0 ; i--){
    perceptron_gHistory[i] = perceptron_gHistory[i-1];
  }
  perceptron_gHistory[0] = outcome;

}