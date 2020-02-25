## Machine Learning based Item State Prediction for openHAB

A first foray into machine learning, and in particular seeing whether it is possible to get usable predictions for openHAB item states. As I have never delved into machine learning before this excercise, it is very simplistic and probably not necassarily the best way of doing such things. Curently only supporting Random Forest and MLP classifiers.



### Requirements:

* Data is retrieved _only_ from an InfluxDB database at the moment.

* Python 3.7 used for the development, along with the following modules:

	- requests
	- sseclient_py==1.7
	- joblib==0.14.1
	- Keras==2.3.1
	- numpy==1.18.1
	- pandas==1.0.1
	- scikit_learn==0.22.1
	- sseclient==0.0.24


* Copy the `openhab_ai.cfg.sample` to `openhab_ai.cfg` and adjust parameters as required.

