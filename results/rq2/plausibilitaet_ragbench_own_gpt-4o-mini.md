# Plausibilitaet: eigene Scores (gpt-4o-mini) gegen RAGBenchs mitgelieferte

| metrik                    | original                  | gruppe   |   n |   spearman_rho |   spearman_p |   kendall_tau |   kendall_p |   median_eigen |   median_original |
|:--------------------------|:--------------------------|:---------|----:|---------------:|-------------:|--------------:|------------:|---------------:|------------------:|
| ragas/faithfulness        | ragas_faithfulness        | alle     | 312 |          0.172 |        0.002 |         0.150 |       0.002 |          0.909 |             1.000 |
| ragas/faithfulness        | ragas_faithfulness        | finqa    | 106 |         -0.059 |        0.549 |        -0.049 |       0.524 |          0.667 |             0.857 |
| ragas/faithfulness        | ragas_faithfulness        | hotpotqa | 111 |          0.243 |        0.010 |         0.226 |       0.010 |          1.000 |             1.000 |
| ragas/faithfulness        | ragas_faithfulness        | techqa   |  95 |          0.158 |        0.127 |         0.139 |       0.114 |          0.900 |             1.000 |
| ragas/context_relevance   | ragas_context_relevance   | alle     | 317 |          0.285 |        0.000 |         0.240 |       0.000 |          1.000 |             0.000 |
| ragas/context_relevance   | ragas_context_relevance   | finqa    | 108 |          0.064 |        0.510 |         0.058 |       0.507 |          1.000 |             0.000 |
| ragas/context_relevance   | ragas_context_relevance   | hotpotqa | 111 |          0.261 |        0.006 |         0.218 |       0.007 |          1.000 |             0.062 |
| ragas/context_relevance   | ragas_context_relevance   | techqa   |  98 |          0.433 |        0.000 |         0.354 |       0.000 |          1.000 |             0.003 |
| trulens/faithfulness      | trulens_groundedness      | alle     | 315 |          0.196 |        0.000 |         0.166 |       0.000 |          1.000 |             0.188 |
| trulens/faithfulness      | trulens_groundedness      | finqa    | 108 |          0.146 |        0.133 |         0.121 |       0.133 |          1.000 |             0.000 |
| trulens/faithfulness      | trulens_groundedness      | hotpotqa | 111 |          0.178 |        0.062 |         0.170 |       0.057 |          1.000 |             0.600 |
| trulens/faithfulness      | trulens_groundedness      | techqa   |  96 |          0.303 |        0.003 |         0.236 |       0.001 |          0.667 |             0.191 |
| trulens/context_relevance | trulens_context_relevance | alle     | 298 |          0.089 |        0.126 |         0.074 |       0.104 |          0.444 |             0.900 |
| trulens/context_relevance | trulens_context_relevance | finqa    |  95 |          0.275 |        0.007 |         0.226 |       0.009 |          0.500 |             0.900 |
| trulens/context_relevance | trulens_context_relevance | hotpotqa | 111 |          0.153 |        0.110 |         0.129 |       0.106 |          0.333 |             1.000 |
| trulens/context_relevance | trulens_context_relevance | techqa   |  92 |          0.552 |        0.000 |         0.451 |       0.000 |          0.467 |             0.800 |
