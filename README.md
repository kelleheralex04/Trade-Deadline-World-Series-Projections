# TradeDeadlineWSProjections
What factors help determine the odds of world series participation the day after the trade deadline?

The dataset for this project is a manually generated from BaseballSavant.MLB.com

Results:
The best formula for predicting which teams will participate in the World Series knowing only data up until the trade deadline (or August 7th):

**run differential/game + all-pitch bullpen velocity + raw away defense + non-HR runs/game**

Run differential is by far the strongest predictor of World Series participation. Among teams with similar overall performance, harder-throwing bullpens and greater non-HR scoring provide additional positive signal. Raw away defensive performance also adds predictive value, although its negative conditional coefficient is difficult to interpret causally.

Because only one team from each league can reach the World Series, we convert the raw participant scores into within-league probabilities. Historical testing favored a moderately smoothed league-normalization (T=1.25), which improved probability calibration and ROC discrimination relative to the raw independent probabilities.

The model selection is based on only seven expanding-window test seasons, so the differences between closely ranked candidate models should be treated as suggestive rather than definitive. More years with Statcast data will allow for more robust modeling.

This model has the highest selection score based on the below model evaluation formula:

Selection Score =
0.20({Both Finalists Top 5}) + 0.13({Top 3}) + 0.10({Top 5}) +
0.15({Average Worst Finalist Rank quality}) +
0.10({Average Finalist Rank quality}) +
0.12({PR AUC quality}) +
0.10({ROC AUC quality}) +
0.10({Brier quality})

Rank	Team	Patricipant Odds	Participant Score
1	MIL	48.38%	0.945
2	BOS	29.83%	0.691
3	LAD	16.67%	0.819
4	DET	14.12%	0.468
5	CWS	13.08%	0.444
6	NYY	10.60%	0.380
7	ATL	8.16%	0.649
8	TB	6.36%	0.245
9	SEA	5.98%	0.231
10	PHI	4.91%	0.495
11	BAL	4.54%	0.175
12	PIT	4.44%	0.463
13	MIN	3.92%	0.150
14	SD	3.92%	0.425
15	CHC	3.59%	0.398
16	HOU	3.01%	0.113
17	CLE	2.79%	0.104
18	TEX	2.59%	0.095
19	ARI	2.41%	0.287
20	MIA	1.76%	0.214
21	SF	1.61%	0.196
22	STL	1.38%	0.168
23	WSH	1.19%	0.143
24	KC	1.19%	0.038
25	LAA	1.06%	0.033
26	TOR	0.69%	0.020
27	COL	0.61%	0.068
28	NYM	0.59%	0.064
29	CIN	0.39%	0.040
30	ATH	0.24%	0.005




