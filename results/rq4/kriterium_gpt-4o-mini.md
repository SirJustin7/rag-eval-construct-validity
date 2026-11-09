# RQ4 — rang-biseriale Korrelation (Cureton 1968), Judge gpt-4o-mini

| metric                     | gruppe            |   n |   anteil_richtig |   r_rb |   r_glass |     p |    P |    Q |   P_max |   distinkte_werte |     u | stufe      | belastbar   |
|:---------------------------|:------------------|----:|-----------------:|-------:|----------:|------:|-----:|-----:|--------:|------------------:|------:|:-----------|:------------|
| deepeval.answer_relevance  | alle              | 200 |            0.645 | -0.250 |    -0.247 | 0.003 | 2654 | 4920 |    9078 |                24 | 0.834 | auswertbar | True        |
| deepeval.answer_relevance  | gpt-4o-2024-05-13 | 100 |            0.820 | -0.259 |    -0.255 | 0.087 |  456 |  833 |    1455 |                21 | 0.834 | auswertbar | True        |
| deepeval.answer_relevance  | llama3:8b         | 100 |            0.470 | -0.140 |    -0.140 | 0.206 |  778 | 1127 |    2488 |                15 | 0.834 | auswertbar | True        |
| deepeval.context_relevance | alle              | 200 |            0.645 | -0.266 |    -0.153 | 0.040 | 2180 | 3577 |    5254 |                57 | 0.600 | auswertbar | True        |
| deepeval.context_relevance | gpt-4o-2024-05-13 | 100 |            0.820 | -0.153 |    -0.069 | 0.601 |  400 |  502 |     666 |                33 | 0.600 | auswertbar | True        |
| deepeval.context_relevance | llama3:8b         | 100 |            0.470 | -0.275 |    -0.216 | 0.032 |  486 | 1025 |    1961 |                33 | 0.600 | auswertbar | True        |
| deepeval.faithfulness      | alle              | 200 |            0.645 |  0.064 |     0.054 | 0.434 | 2548 | 2051 |    7740 |                12 | 0.494 | auswertbar | True        |
| deepeval.faithfulness      | gpt-4o-2024-05-13 | 100 |            0.820 | -0.044 |    -0.043 | 0.730 |  335 |  399 |    1462 |                11 | 0.494 | auswertbar | True        |
| deepeval.faithfulness      | llama3:8b         | 100 |            0.470 |  0.199 |     0.109 | 0.244 |  725 |  454 |    1363 |                 8 | 0.494 | auswertbar | True        |
| ragas.answer_relevance     | alle              | 200 |            0.645 |  0.056 |     0.056 | 0.516 | 4783 | 4274 |    9159 |               173 | 0.985 | auswertbar | True        |
| ragas.answer_relevance     | gpt-4o-2024-05-13 | 100 |            0.820 | -0.016 |    -0.016 | 0.918 |  726 |  750 |    1476 |                99 | 0.985 | auswertbar | True        |
| ragas.answer_relevance     | llama3:8b         | 100 |            0.470 | -0.158 |    -0.158 | 0.174 | 1004 | 1397 |    2491 |                81 | 0.985 | auswertbar | True        |
| ragas.context_relevance    | alle              | 200 |            0.645 | -0.172 |    -0.166 | 0.044 | 2666 | 4183 |    8835 |                 5 | 0.745 | auswertbar | True        |
| ragas.context_relevance    | gpt-4o-2024-05-13 | 100 |            0.820 | -0.195 |    -0.190 | 0.193 |  443 |  724 |    1440 |                 5 | 0.745 | auswertbar | True        |
| ragas.context_relevance    | llama3:8b         | 100 |            0.470 | -0.241 |    -0.208 | 0.061 |  707 | 1226 |    2155 |                 5 | 0.745 | auswertbar | True        |
| ragas.faithfulness         | alle              | 200 |            0.645 |  0.053 |     0.052 | 0.533 | 4077 | 3601 |    8946 |                32 | 0.840 | auswertbar | True        |
| ragas.faithfulness         | gpt-4o-2024-05-13 | 100 |            0.820 | -0.179 |    -0.157 | 0.294 |  555 |  787 |    1296 |                29 | 0.840 | auswertbar | True        |
| ragas.faithfulness         | llama3:8b         | 100 |            0.470 | -0.015 |    -0.015 | 0.893 |  930 |  968 |    2488 |                17 | 0.840 | auswertbar | True        |
| trulens.answer_relevance   | alle              | 200 |            0.645 |  0.471 |     0.382 | 0.000 | 4918 | 1417 |    7431 |                 4 | 0.655 | auswertbar | True        |
| trulens.answer_relevance   | gpt-4o-2024-05-13 | 100 |            0.820 |  0.257 |     0.195 | 0.129 |  530 |  242 |    1119 |                 3 | 0.655 | auswertbar | True        |
| trulens.answer_relevance   | llama3:8b         | 100 |            0.470 |  0.150 |     0.124 | 0.238 |  896 |  587 |    2061 |                 4 | 0.655 | auswertbar | True        |
| trulens.context_relevance  | alle              | 200 |            0.645 | -0.170 |    -0.168 | 0.048 | 3340 | 4881 |    9089 |                13 | 0.894 | auswertbar | True        |
| trulens.context_relevance  | gpt-4o-2024-05-13 | 100 |            0.820 | -0.077 |    -0.076 | 0.616 |  617 |  729 |    1452 |                12 | 0.894 | auswertbar | True        |
| trulens.context_relevance  | llama3:8b         | 100 |            0.470 | -0.292 |    -0.290 | 0.012 |  757 | 1479 |    2471 |                12 | 0.894 | auswertbar | True        |
| trulens.faithfulness       | alle              | 200 |            0.645 |  0.063 |     0.063 | 0.460 | 4356 | 3783 |    9151 |                35 | 0.891 | auswertbar | True        |
| trulens.faithfulness       | gpt-4o-2024-05-13 | 100 |            0.820 | -0.280 |    -0.276 | 0.067 |  488 |  896 |    1458 |                31 | 0.891 | auswertbar | True        |
| trulens.faithfulness       | llama3:8b         | 100 |            0.470 |  0.185 |     0.184 | 0.106 | 1271 |  813 |    2473 |                22 | 0.891 | auswertbar | True        |
