# Real-Time-Filter-GUI-
Digital filter - uses python tkinter framework 
This project report entitled to “Real Time Filter.” The main objective of the filter application is to filter the sound signal coming from the mic (earphones mic basically) using the response of the filter given by the user as input.
Design of this filter is done by python framework Tkinter and other important libraries in python responsible for any kind of scientific computation wherever needed. The code does the filtering by not using a direct filter library in python but from scratch.

***********************************************************************************************************************************************************************************
Objectives acheived-
The filter uses the python framework Tkinter for interacting with the user. The user inputs the format of filter system as in the format s/he has. Then he chooses the type of filter (LP, HP, BP, BS) he wants as the output. S/he enters the cutoff frequency/frequencies in the dialog boxes and clicks on the process button. The backend processes the information given by the user and allows s/he to record the voice. Then the backend listens to the voice and processes it to give the filtered output as desired by the user. At this stage the user can choose to either record and save the output file or record and listen to it immediately. There is no use of any direct filter in the python libraries available, instead the author has tried to make use of the basic concepts used in class to design the filter like
taking DTFT, convolution, IDTFT etc.
