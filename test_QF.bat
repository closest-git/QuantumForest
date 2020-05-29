::@echo off
E:
cd  E:\fengnaixing\cys\QuantumForest
:: https://stackoverflow.com/questions/18462169/how-to-loop-through-array-in-batch
::set DATA[0]=YAHOO
::set DATA[1]=CLICK
::set DATA[2]=MICROSOFT
::set DATA[3]=YEAR
::set DATA[4]=HIGGS
::set DATA[5]=EPSILON
set "cmd=python main_tabular_data.py --data_root=../Datasets/"
::set "param=--model=GBDT"

%cmd% --dataset=YAHOO %param%
%cmd% --dataset=CLICK %param%
%cmd% --dataset=MICROSOFT %param%
%cmd% --dataset=YEAR %param%
%cmd% --dataset=HIGGS %param%
%cmd% --dataset=EPSILON %param%
:: --model=GBDT --dataset=MICROSOFT --learning_rate=0.001
:: --attention=""
:: --scale="large"
:: C:/Users/fengnaixing/test_QF.bat
:: python main_tabular_data.py --data_root=../Datasets/ --dataset=YAHOO --model=GBDT