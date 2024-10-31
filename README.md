# MosaicPineapple Project: English Proficiency Detection and Generation

This project uses a dataset of CEFR labeled English texts to train discriminative language models to be able to detect the CEFR level of a new input text. We will then use data gathered concerning the effectiveness of language models at distinguishing between closely related CEFR levels (i.e. B1 and B2) compared to further apart levels (i.e. A1 and C1) to evaluate the effectiveness of automatic proficiency detection. Additionally, the various models will be compared directly to see if any models performed exceptionally well in certain scenarios, and we may use prompting techniques or fine tuning to achieve better results.

Dataset Source:
CEFR-SP (https://github.com/yukiar/CEFR-SP), a dataset of about 10000 publicly accessible English sentences labeled by their CEFR proficiency level, made for the following paper:

Yuki Arase, Satoru Uchida, and Tomoyuki Kajiwara. 2022. CEFR-Based Sentence-Difficulty Annotation and Assessment. 
in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2022) (Dec. 2022).
