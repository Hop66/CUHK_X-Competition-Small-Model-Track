# A Large-Scale Multimodal Dataset and Benchmarks for Human Activity Scene Understanding and Reasoning

Siyang Jiang<sup>1</sup>, Mu Yuan<sup>1</sup>, Xiang Ji<sup>1</sup>, Bufang Yang<sup>1</sup> Zeyu Liu<sup>2</sup>, Lilin Xu<sup>3</sup>, Yang Li<sup>1</sup>, Yuting He<sup>1</sup>, Liran Dong<sup>1</sup>, Wenrui Lu<sup>1</sup> Zhenyu Yan<sup>1</sup>, Xiaofan Jiang<sup>3</sup>, Wei Gao<sup>4</sup>, Hongkai Chen<sup>1,</sup>, Guoliang Xing<sup>1,</sup> <sup>1</sup>The Chinese University of Hong Kong, Hong Kong, <sup>2</sup>University of Illinois Urbana-Champaign, United States, <sup>3</sup>Columbia University, United States, <sup>4</sup>University of Pittsburgh, United States.

## ABSTRACT

Multimodal human action recognition (HAR) utilizes comple mentary data for activity classification. Built on traditional HAR tasks, recent advances in Large Language Models (LLMs) enable detailed descriptions and causal reasoning of human actions, advancing new tasks of human action understand ing (HAU) and human action reasoning (HARn). However, most LLMs, especially multimodal Large Vision-Language Models (LVLMs), struggle with modalities other than RGB images, like depth, IMU, or mmWave, due to a lack of largescale <data,caption> datasets in these task domains. Existing HAR datasets provide only coarse-grained <data,label> annotations, insuficient for depicting the detailed action dy namics required in HAU and HAR tasks.<sup>1</sup> Simply combining annotations and generating captions with LLMs often lacks the necessary logical and spatiotemporal consistency.

In this paper, we introduce CUHK-X, a large-scale multimodal dataset and benchmarks for HAR, HAU, and HARn. It includes 58,445 samples of40 actions performed by 30 partici pants across two indoor environments, covering diverse daily scenarios. To address the challenge of spatiotemporal inconsistencies in captions, we propose a prompt-based scene creation method that leverages LLMs to generate logically connected activity sequences. CUHK-X also includes three benchmarks with six tasks to evaluate state-of-the-art models. Experimen tal results show average accuracies of76.52% for HAR, 40.76% for HAU, and 70.25% for HARn. This large-scale multimodal dataset aims to empower the research community to apply, develop, and adapt data-intensive learning techniques for a wide range of human activity-related tasks. The project page and code is available at https://openaiotlab.github.io/CUHK-X and https://github.com/openaiotlab/CUHK-X, respectively.

## 1 INTRODUCTION

In recent years, human action recognition (HAR) tasks have advanced significantly, leveraging artificial intelligence to classify human activities from multimodal sensory data [36, 64]. Beyond classification, Human Action Understanding (HAU) and Human Action Reasoning (HARn) tasks provide richer and more detailed descriptions of human activities, enabling diverse applications across various domains such as healthcare, daily living assistance, and surveillance [22, 54, 56]. For example, in the management of Alzheimer’s disease, a coherent understanding of a patient’s longitudinal behaviors is crucial for monitoring daily routines, providing timely caregiver support, and preventing accidents [38]. As shown in Fig. 1, a traditional HAR task is limited to recognizing isolated human actions, such as “sleep” or “fall”, and lacks the ability to interpret a continuous sequence of actions. In contrast, the HAU task addresses this limitation by understanding and providing natural language descriptions of the sequence of actions. For example, describing a scene as “the subject is eating, holding a knife in his right hand and a fork in his left”

![](images/c09dc96f11fae375a587aada3662c7f6e4532bf566dfd9afa8256b9e42b271ca.jpg)  
Figure 1: CUHK-X captures a multi-room home environment and supports three tasks: HAR (classification), HAU (captioning), and HARn (intention prediction). It integrates diverse modalities, including RGB, depth, thermal, infrared, IMU, skeleton, and mmWave.

provides valuable context for the early detection of cognitive decline. Furthermore, the HARn task infers intentions from sequences of human actions and predicts future actions. A typical example is observing “a subject is walking toward to sofa”; subsequent action might be predicted as “attempting to watch television or sit down”, thereby triggering a preventative intervention.

In practice, human action understanding and intention pre diction can rarely done by a straightforward autoregressive process with conventional deep neural networks (DNNs). Instead, it requires capability for knowledge representation and logical reasoning that integrate environmental contexts and scene knowledge [57]. To obtain these capabilities, existing techniques usually fine-tune Large Language Models (LLMs) using high-quality datasets annotated as <data,caption> pairs. Furthermore, logical reasoning in LLMs can be explicitly elicited through methodologies such as Chain-of-Thought [33], Tree-of-Thought [59], or Graph-of-Thought [43].

However, most existing HAR datasets provide only coarsegrained <data,label> pairs for RGB images [52], thus unsuitable for fine-tuning LLMs for HAU and HARn tasks. Although somerecentdatasetsoferfine-grained<data,caption>pairs for RGB images [5, 16, 17], the fixed fields ofviews (FOVs) and limited mobility of RGB cameras hinder the timely capture of human behavior in many practical scenarios. Moreover, in privacy-sensitive scenarios such as daily home monitoring, RGB images pose risks by potentially containing sensitive personal data. Consequently, alternative sensor modalities, such as depth, thermal, IMU, and mmWave, are preferable.

Most of current LLMs, particularly Large Vision Language Models (LVLMs), perform robustly with RGB and textual data, but they encounter significant dificulties when ap plied to other prevalent non-RGB modalities. The primary reason is the notable scarcity of large-scale datasets with <data,caption> pairs in non-RGB modalities [23], as most existing ones are confined to coarse-grained <data,label> annotations [31, 42, 65]. A naive approach to obtaining captioned datasets across multiple sensory modalities involves merging unimodal datasets, combining their coarse-grained labels, and using an LLM to generate captions. However, this method frequently yields captions that lack the essential spa tiotemporal consistency. For example, directly combining actions like brushing teeth and eating into a single scene is illogical, as these two actions typically occur independently in distinct contexts (e.g., a bathroom versus a dining room). Additionally, when generating captions, LLMs often fail to accurately infer humans’ behavioral contexts from the given coarse-grained labels, due to their limited representational capacity. This often results in incomplete, inaccurate, or even misleading captions (see §A.2 for details).

To address these gaps, we present CUHK-X, a large-scale multimodal dataset with seven synchronized modalities (RGB, depth, infrared (IR), thermal, skeleton, IMU, and mmWave) and three benchmarks for HAU and HARn tasks, while also supporting conventional HAR tasks. The CUHK-X dataset employs a ground-truth-first (GT-first) data collection scheme, where target states are predefined before data recording, to ensure the acquisition of precisely aligned multimodal signals. To prevent spatiotemporal inconsistencies and guarantee accurate ground truth, the construction ofCUHK-X begins with a Scene-based Caption Generation Framework. This framework categorizes human actions into seven thematic groups based on the American Time Use Survey (ATUS) [34, 61], from which 40 representative actions are carefully selected based on their frequency and relevance in prior benchmark datasets (e.g., HHAR [46], UCI [41], and Cosmo [36]). Subsequently, LLMs are used to logically connect these actions into semantically coherent captions that depict predefined scenes of daily living, such as living rooms, kitchens, bedrooms, and bathrooms. These captions incorporate varied contexts (i.e., performing actions in a relaxed, calm or hurried manner) to further enrich the narrative coherence. Lastly, we employ a human-checking stage to ensure that the generated captions are consistent with the ground truth, by validating physical plausibility and temporal logic. (see overview in §A.1.)

Using the generated captions as the ground truth, CUHK-X comprises over 58,445 daily activity samples from 30 participants, captured across two indoor settings using seven modalities, including RGB, depth, thermal, infrared, skeleton, IMU, and mmWave sensors. The sensor suite includes a Goermicro Vzense NYX 650 (depth), Texas Instruments IWR6843ISK (mmWave radar), Hikvision TB4117 (thermal), and five Wit-Motion WT 9011DCL-BT50 IMUs. Participants are instructed to understand and act out the generated captions, enabling the collection of high-quality, well-aligned data pairs.

To verify the dataset’s utility, we provide benchmarks for six tasks spanning HAR, HAU, and HARn, by evaluating stateof-the-art baselines of DNNs and LLMs to these benchmark tasks. These tasks include (1) HAR; (2-5) HAU tasks (caption comparison, context analysis, sequential action reordering, and action selection); and (6) HARn. The HAR benchmark val idates the dataset’s suficient knowledge for recognition tasks. The HAU benchmarks assess caption comparison (against ground truth), context analysis (e.g., inferring speed ofaction), temporal ordering (for shufled actions), and action identification (from a predefined set). The HARn benchmark evaluates an LLM’s ability to infer intentions, causal relationships, and logical action progression. For HAR evaluation, we used state-of-the-art recognition models for each sensor modality, and analyze model performance under long-tail distributions and cross subject situations. For HAU and HARn evaluation, we employed state-of-the-art baselines, including four captioning models (InternVL2.5-2B [10], InternVL2.5-8B [10], QwenVL2.5-3B [3], QwenVL2.5-7B [3]) and two reasoning models (VideoLLaVA-7B [28] and VideoChatR1-7B [27]). The goal of these benchmarks is to explore the tasks performance and diferentiability over diferent models and modalities (see §A.1 for overview illustration).

Experimental results demonstrate that fine-tuning models on CUHK-X significantly improves HAR accuracy compared to using pre-trained models alone, confirming that the dataset provides the necessary knowledge. Specifically, we achieve an average accuracy of 76.52% across seven modalities for HAR. Additionally, we achieve an average accuracy of 40.76% (max 50.52%) across all HAU tasks. Moreover, we achieve an average accuracy of70.25% (max 90.30%) across three vision modalities for HARn. These results confirm that CUHK-X enables robust benchmarking and bridges the key gaps in existing datasets. The main contributions are summarized as follows:

• We introduce CUHK-X, a large-scale multimodal dataset comprising 58,445 samples collected from 30 participants using seven sensor modalities (RGB, depth, thermal, Infrared, IMU, mmWave, and skeleton) across two real-world environments. It provides diverse and realistic activity data and captions for advanced research in HAR, HAU, and HARn.

• To address challenges in logical consistency and spatiotemporal representation, we propose a prompt-based scene creation that leverages prompt-driven LLMs to generate logically connected actions in daily activity scenes with a human checking stage.

• We establish three benchmarks containing six tasks to systematically evaluate state-of-the-art baselines. Through rigorous analysis of tasks performance and diferentiability over diferent models and modalities, we position CUHK-X as a cornerstone dataset for advancing research in HAR, HAU, and HARn.

## 2 MOTIVATION STUDY

This section outlines the tasks ofHAU and HARn, followed by a discussion on the limitations of existing datasets in supporting these tasks.

## 2.1 Applications of HAU and HARn

The CUHK-X dataset can be applied to HAU and HARn tasks across various domains, including smart health [57], smart home [8], and disease intervention [35]. It enables continuous, longitudinal monitoring and analysis ofuser behavior, such as in Alzheimer’s Disease monitoring [9], Parkinson’s Disease management [47]. Beyond healthcare, CUHK-X can also sup port smart home systems by enhancing comfort and energy eficiency. For example, it can use the predicted user actions to adjust room lighting or thermostat settings, thereby optimizing energy consumption while maintaining a comfortable environment. This enriched holistic understanding, facilitated by CUHK-X, is critical not only for improved caregiving but also for creating smarter and more eficient living spaces.

## 2.2 Limitations of Existing Datasets

Existing multimodal datasets often sufer from key limitations, such as small subject pools (e.g., USC [65], Shoaib [44], HHAR [46]) and a restricted range of activities (e.g., UTD [7], mRI [1], Thermal-IM [49]). While datasets like NTU-RGBD [31] and Ego-Exo4D [16] are more extensive, they frequently lack modality diversity (e.g., most miss IR, thermal, or captions), as summarized in Table 2.

2.2.1 Limitations ofexisting coarse-grainedHAR datasets. As shown in Table 2, many earlier datasets, such as USC [65], Shoaib [44], and HHAR [46], are constrained by small participant numbers (fewer than 15) and a narrow range of activities (e.g., only 6–12 actions). Similarly, datasets like Thermal-IM [49] and UTD [7] involve too few participants or a limited number of activities. Some recent datasets, such as NTU-60/120 [31, 42], and PKU-MMD [11], include larger participant cohorts (e.g., 66 people in PKU-MMD) and more activity classes (e.g., 60 actions in NTU-60). However, they primarily focus on RGB and skeleton data, overlooking essential modalities like thermal, IR, and IMUs. For instance, action recognition relying solely on RGB data becomes challenging under occlusion or when a person faces away from the camera. Thus, a major limitation of coarse-grained datasets is their inability to provide suficient detail for HAU or HARn tasks.

2.2.2 Limitations ofexistingfine-grained HAU datasets. Previous fine-grained datasets, such as Ego-4D [16] and Ego-Exo4D [17], enhance model capabilities for detailed human action understanding by providing rich data descriptions. Nonetheless, these datasets are limited in modality coverage. As shown in Table 2, Ego-4D and Ego-Exo4D include RGB data but lack other critical modalities, such as depth, thermal, infrared, and skeleton. Moreover, state-of-the-art captioning models, such as Tarsier [53] and Tarsier2 [62], are designed specifically for RGB data, leading to suboptimal performance in other modalities. To illustrate, as shown in Fig. 2, we conducted experiments on UTD [7], Thermal-IM [49], and PKU-MMD [11] datasets using Tarsier2 (see §A.2 for details).

## 2.3 Summary

In summary, existing coarse-grained datasets are unsuitable for HAU and HARn tasks due to insuficient descriptive detail, while existing fine-grained datasets fail to comprehensively cover multiple modalities. To address these gaps, CUHK-X provides a wider variety of modalities (e.g., RGB, depth, thermal, IR, skeleton, IMU, and captions), a more diverse range of activities, and a larger participant cohort. This comprehensive

![](images/d67c4fc732d182c1799b2dabd8cd15c73ab88c113f2b101efa016e6332bab4a7.jpg)

A person stands before a whiteboard.. then extends their arms forward with palms facing outward, and finally lowers their hands back to their sides.

A person is walking forward against a black ..., and the person then lowers their arms and continues walking forward. Inaccurate Actions

A man in a red shirt and gray pants walks towards a .... He opens the bottle and takes a drink. After drinking, he places the bottle back on the table.

A man enters the room from the left side and walks towards a table with chairs, .... He opens the bottle and places it on the table. Missing Actions

A man stands,... picks up a pen, and writes something on a piece of paper. After writing, he walks back to the center of the room, claps his hands, and stands still.

(c) PKU-MMD Dataset

Figure 2: Limitations of SOTA LVLMs in HAU tasks.

dataset not only supports HAR tasks but also enables finegrained HAU tasks that demand a deeper contextual under standing, as well as reasoning-based tasks like HARn, which require understanding sequential actions and predicting future behaviors. Most importantly, it establishes a foundational resource for developing robust multimodal systems for sophisticated human behavior analysis in real-world scenarios.

## 3 GROUND TRUTH-FIRST

## DATA COLLECTION STRATEGY

This section describes our ground truth-first (GT-first) methodology for human action data collection, positioning it as a practically better alternative compared to conventional data-first approaches to human data collection.

## 3.1 Motivation for a GT-First Strategy

Data collection and annotation typically follow two paradigms: data-first [25] and ground-truth-first, or GT-first [1, 5, 31, 36, 58]. Data-first strategy, as the most intuitive method of data collection, first gathers large-scale real-world data from hu man subjects, and then apply annotations to the collected data afterwards. While conceptually reasonable, these strategies present significant limitations in practice. In particular, since data collection is conducted before annotations and hence lacks specific guidelines, the collected data requires intensive postprocessing and filtering to ensure eficient and proper annotation. These extra eforts make data collection highly inefective and not scalable, especially when involving large groups ofhuman subjects. In addition, since annotations have to be applied to collected data, it is expensive to fix any early mistakes in the human action set, raising extra privacy and consent risks when recording people and homes.

To avoid these limitations, most existing large-scale datasets, such as ImageNet [14] and NTU-RGBD [31], follow the GTfirst strategy for data collection. Being diferent from the datafirst strategy, the GT-first strategy defines the label space, annotation rules, and target scenes in advance, and then collects data based on those definitions. Prioritizing ground truth yields clear benefits: 1) focus, as only planned cases are recorded; 2) eficiency, as annotators verify checklists rather than create labels from scratch; 3) scalability, as the same collection scripts and annotation rules can be reused across sites.

However, the approach would introduce bias compared to the data-first approach. In the following, our objective is to analyze and mitigate three inherent biases ofGT-first approaches in CUHK-X, i.e., coverage, diversity, and discrepancy, through several targeted strategies including: (1) refining class coverage within defined boundaries, (2) enhancing variation via intra- and inter-class combinations coupled with linguistic enrichment, and (3) incorporating human-in-theloop verification to ensure physical and logical coherence. These methodological components are elaborated in §4.

## 3.2 Bias in GT-First Data Collection

GT-first approaches, i.e., those that derive supervision or text directly from predefined actions and associated templates, ofer clear benefits in controllability and reproducibility to obtain the GT. In CUHK-X, we adopt GT-first approach to obtain the ground truth. However, it also induces systematic artifacts that reflect the prior encoded by the ontology and the templating process. In particular, they may bias models toward (i) Coverage, (ii) Diversity and (iii) Discrepancy. In the following, we justify the biases in CUHK-X.

3.2.1 Coverage. We define coverage as the number of classes included in the dataset. It is hard for any dataset to include all human actions. In practice, some labels mean more than one thing, and they are not consistent across datasets. Consequently, in CUHK-X, we adopt a closed-world objective, i.e., we focus on a fine-grained subset of actions curated from prior research and informed by our experimental evidence. In particular, we coarse-grained our action selection using ATUS [34] and the action frequency across datasets (§4.1.1), and then fine-grained the selection of several significant actions based on prior studies (§4.1.2).

Table 1: Evaluation on discrepancy.

<table><tr><td>BertScore-F1</td><td>Com-Cap.</td><td>Free-Act.</td><td>Instruct-Act.</td></tr><tr><td>Task-1</td><td>93.40%</td><td>95.50%</td><td>93.60%</td></tr><tr><td>Task-2</td><td>92.90%</td><td>94.10%</td><td>93.10%</td></tr><tr><td>Task-3</td><td>93.50%</td><td>94.20%</td><td>92.70%</td></tr><tr><td>Task-4</td><td>92.20%</td><td>93.80%</td><td>92.50%</td></tr></table>

3.2.2 Diversity. Even when labels are well covered, their realizations can exhibit narrow lexicalization, limited con textual variety, and stereotyped co-occurrence patterns. We therefore define diversity, for each modality, as the extent to which samples within the same class difer from one another. For example, in each modality, instances of the class “drinking water” should vary in setting, poses, and previous or co-occurring actions. To mitigate this limitation, we first generate captions from diverse intra- and inter-class action combinations (§4.2.1) and then enrich those captions with linguistic variation to diversify lexical choice, syntax, and contextual framing (§4.2.2).

3.2.3 Discrepancy. In CUHK-X, discrepancy is denoted from two perspectives. Firstly, it is the mismatch between captions generated by an LLM and those authored by humans for the same meaning. To minimize this discrepancy, we incorporate a human checking stage that enforces physical plausibility, logical coherence, and contextual appropriateness (§4.2.3), aiming to bring LLM captions close to human parity. Secondly, we consider whether the data reflect key actions performed naturally versus performed by following explicit instructions. Because CUHK-X focuses on daily activities, we regard any bias introduced by instruction-following as acceptable. In practice, we asked the volunteer to read the instructions first, then perform the action based on their understanding. We also conduct an experiment to assess this efect.

Specifically, we set a home ofice setting with four microtasks: drinking water (Task-1), listening to music with earphones (Task-2), writing notes (Task-3), and answering a phone call (Task-4). For each task, we recruited three volunteers (10% ofthe participants in CUHK-X) to act freely and an notate their recordings. Next, the same volunteers performed the actions following instructions derived from their free ac tions. We again collected three independent captions. We compared the semantic similarity of captions using close-source LLMs, i.e., QwenVL3-235B, between the “free” and “instructed” conditions using BertScore. Across all four tasks, the two sets of captions are highly similar, indicating that both conditions efectively capture the underlying action information (Table 1, Com-Cap.). Furthermore, when compared the generated captions against human-annotated ground truth, the results are also accurate, as shown in Table 1 (Free-Act. and Instruct-Act.). These findings justify our focus on human actions, and the remaining discrepancies are acceptable within CUHK-X.

![](images/d6ec0bec9c5fe91dcff169f16f3243ee83bf7a66a27d3853275ffaa5e58571c4.jpg)  
Figure 3: Frequency of the actions among USC [65], Shoaib [44], HHAR [46], UTD [7] ActivityNet [61], UCI [41], NTU [31, 42], PKU-MMD [11], Cosmo [36], mRI [1], Thermal-IM [49].

## 4 SCENE-BASED CAPTION GENERATION

## 4.1 Prior knowledge-based Action Selection

CUHK-X is developed through a labor-intensive collection of multimodal data in real-world scenarios. In this section, we describe how CUHK-X incorporates typical daily actions with a two-stage selection process to select representative actions.

4.1.1 Coarse-grained Action Selections via Predefined Categories and Cross-dataset Frequency. Firstly, based on the ATUS [34] activity hierarchy and ActivityNet [5], we categorized the activity classes into seven top-level categories: Personal Care, Eating and Drinking, Household, Caring and Helping, Working, Socializing and Leisure, and Sports and Exercise. CUHK-X adopts a structured semantic framework that leverages hierarchical relationships between activities, ensuring the selection of typical and comprehensive real-world daily activities. Next, we analyzed the action frequency in 12 popular human action recognition datasets, such as NTU [31, 42], UTD [7], and UCI [41], which are primarily summarized in Table 2. As illustrated in Fig. 3, we analyze the high-frequency occurrences within these datasets. Specifically, the total number of action classes is 349, which are consolidated into 127 classes by merging those with similar meanings. However, since most actions appear only once, we focus exclusively on high-frequency actions (#frequency > 4) for further analysis.

4.1.2 Fine-grained Action Selections via Prior Studies. Then, we carefully selected fine-grained, representative actions based on insights from previous research. Specifically, the Personal Care category (6 actions) was guided by the prior study [32]. The Eating and Drinking (6 actions) and Household (5 actions) categories were guided by findings from the previous work [40]. The Working category (6 actions) was inspired by [2], while the Socializing and Leisure category (5 actions) was shaped by [4]. Finally, the Sports and Exercises (9 actions) and Caring and Helping (3 actions) categories were supported by insights from Gerber et al. [15].

As shown in Fig. 4, we select 40 actions which are divided into seven categories include the following: (1) Personal Care, which has 6 actions including Washing face, Brushing teeth, Combing hair, Undressing, Wiping hands, and Getting

Table 2: A summary of the related coarse-grained HAR and fine-grained HAU datasets ( indicates inclusion). DEP, THE, IR, SKE denotes depth, thermal, infrared, skeleton modalities, respectively.

<table><tr><td>Dataset</td><td>Years</td><td># of Samples</td><td># of Subjects</td><td># of Activities</td><td>RGB</td><td>DEP</td><td>THE</td><td>IR</td><td>SKE</td><td>IMU</td><td>mmWave</td><td>Caption</td></tr><tr><td>USC [65]</td><td>2012</td><td>840</td><td>14</td><td>12</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td><td>○</td><td>○</td></tr><tr><td>Shoaib [44]</td><td>2014</td><td>70</td><td>10</td><td>7</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td><td>○</td><td>○</td></tr><tr><td>HHAR [46]</td><td>2015</td><td>3,240</td><td>9</td><td>6</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td><td>○</td><td>○</td></tr><tr><td>UTD [7]</td><td>2015</td><td>3,444</td><td>8</td><td>27</td><td>●</td><td>●</td><td>○</td><td>○</td><td>●</td><td>●</td><td>○</td><td>○</td></tr><tr><td>UCI [41]</td><td>2016</td><td>180</td><td>30</td><td>6</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td><td>○</td><td>○</td></tr><tr><td>NTU-60 [42]</td><td>2016</td><td>56,880</td><td>40</td><td>60</td><td>●</td><td>●</td><td>○</td><td>●</td><td>●</td><td>○</td><td>○</td><td>○</td></tr><tr><td>PKU-MMD [11]</td><td>2017</td><td>20,000</td><td>66</td><td>51</td><td>●</td><td>●</td><td>○</td><td>●</td><td>●</td><td>○</td><td>○</td><td>○</td></tr><tr><td>NTU-120 [31]</td><td>2019</td><td>114,480</td><td>106</td><td>120</td><td>●</td><td>●</td><td>○</td><td>●</td><td>●</td><td>○</td><td>○</td><td>○</td></tr><tr><td>Cosmo [36]</td><td>2022</td><td>3,434</td><td>30</td><td>14</td><td>○</td><td>●</td><td>○</td><td>○</td><td>○</td><td>●</td><td>●</td><td>○</td></tr><tr><td>mRI [1]</td><td>2022</td><td>300</td><td>20</td><td>12</td><td>●</td><td>●</td><td>○</td><td>○</td><td>○</td><td>●</td><td>●</td><td>○</td></tr><tr><td>Thermal-IM [50]</td><td>2023</td><td>783</td><td>2</td><td>24</td><td>●</td><td>○</td><td>●</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>MM-Fi [58]</td><td>2023</td><td>1080</td><td>40</td><td>27</td><td>●</td><td>●</td><td>○</td><td>○</td><td>●</td><td>○</td><td>●</td><td>○</td></tr><tr><td>XRF55 [52]</td><td>2024</td><td>42,900</td><td>39</td><td>55</td><td>●</td><td>●</td><td>○</td><td>●</td><td>○</td><td>○</td><td>●</td><td>○</td></tr><tr><td>ActivityNet [5]</td><td>2015</td><td>9,682</td><td>-</td><td>203</td><td>●</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td></tr><tr><td>Ego-4D [16]</td><td>2022</td><td>5,831</td><td>923</td><td>146</td><td>●</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td><td>○</td><td>●</td></tr><tr><td>Ego-Exo4D [17]</td><td>2024</td><td>5,035</td><td>740</td><td>689</td><td>●</td><td>○</td><td>○</td><td>○</td><td>○</td><td>●</td><td>○</td><td>●</td></tr><tr><td>CUHK-X</td><td>2025</td><td>58,445</td><td>30</td><td>40</td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td><td>●</td></tr></table>

Dressed; (2) Eating and Drinking, which has 6 actions including Drinking, Eating, Grabbing utensils, Pouring, Stirring, and Peeling fruit; (3) Household, with 5 actions including Sweeping, Mopping, Washing dishes, Wiping surface, and Folding clothes; (4) Working, which includes 6 actions includ ing Typing on a keyboard, Writing, Calling, Checking the time, Reading and Turning a page; (5) Socializing and Leisure, with 5 actions including Taking a selfie, Playing board games, Watching TV, Using a phone, and Listening to the music with headphones; (6) Sports and Exercises, which has 9 actions including Walking, Lunges, Siting down, Lying down, Standing up, stretching,Jumpingjacks,Squats and Runningand (7) Caring and Helping, with 3 actions including Taking medicine, Checking body temperature, and Massaging oneself.

## 4.2 Prompt-based Scene

## Creation with Human-driven Checking

In this subsection, we propose a prompt-based scene creation approach designed to logically connect the selected actions (§4.1) via constructing various daily living scenes.

4.2.1 Intra- and Inter-categories Caption Generation. Our goal is to connect as many selected actions as possible into a coherent and logical sentence that aligns with everyday life scenarios. To achieve this, we implement a two-stage prompt design primarily based on the selected actions, ensuring both diversity and relevance in the generated captions. Specifically, we design the prompt to encourage the LLM to combine multiple actions into a cohesive scene within each category. For instance, actions such as “Washing face,” “Brushing teeth,”

“Combing hair,” “Dressing,” and “Wiping hands” can be combined to generate a detailed scene: The user wakes up, opens the curtains, and stretches (Stretching). The user walks to the bathroom and washes theirface (Washingface) with water or facial cleanser, then dries it with a towel. The user picks up a toothbrush, squeezes toothpaste, and begins brushing theirteeth (Brushing teeth). After brushing, they rinse their mouth with waterandclean the toothbrush. The useruses a comb to carefully brush their hair (Brushing hair), possibly tying it up or styling it. The user quickly wipes their hands (Wiping hands) with a towel or tissue. Finally, the user returns to the bedroom, selects clothesfrom the wardrobe, and completes the process ofgetting dressed (Dressing). Similarly, actions from other categories are combined via LLMs to create contextually rich captions that reflect realistic and meaningful daily scenarios. This method ensures that the generated captions not only integrate mul tiple actions logically but also create a natural flow of events that mirrors real-life activity patterns.

4.2.2 Enriching Captions through Language Diversity. To further enhance the diversity of captions, we leverage LLMs to enrich them by expanding or substituting their sentence components. Specifically, a sentence is composed of several key elements, including the subject, predicate, object, attribute, adverbial, and complement. In the context of central actions in human action understanding, the subject, predicate, and object are typically predefined. Thus, we use LLMs such as GPT-4o [19] or DeepSeek [29], to add more attributes and adverbials. For instance, we can enrich the description of the example in §4.2.1 by incorporating adverbs before the verb to provide additional context and nuance. Specifically, instead of a straightforward description, the user can carefully squeeze a small amount of toothpaste onto the bristles and begin brushing their teeth (Brushing teeth), with added detail such as “quickly,” “smoothly,” or “slowly” to describe how the action is performed. By enriching the attributes and adverbials, the generated captions provide a more detailed and vivid depiction of actions, creating a natural flow between individual activities. This level of detail not only enhances the linguistic diversity of captions but also improves their utility in datasets for tasks such as human action understanding and multimodal learning.

![](images/38200ced90c7b06e86aafabf74cf0faa39fe47e1bb6d7a7d3224376fd376f74a.jpg)  
Figure 4: CUHK-X includes 40 actions in 7 categories.

4.2.3 Human Checking via Physical Knowledge and Logical Coherence. To reduce discrepancy and hallucination in LLM generated captions, we implement a human checking verification stage that enforces physical plausibility, scene logic, and dataset conventions before acceptance as ground truth. Four graduate-level raters conduct this review using a structured checklist and edit protocol.

We validate captions against the following criteria: (1) Physical feasibility and kinematics. Body-pose and object-state transitions must obey continuity. For example, “cup is empty” → “pouring into cup” → “cup becomes full,” not the reverse. (2) Scene and environment consistency. Actions and objects must be compatible with the room and floor plan (e.g., “brush ing teeth” in a bathroom; “watching TV” requires a visible or plausibly placed TV). Captions must not assert observations outside a sensor’s FOV. (3) Temporal and causal coherence. Event order must be logically progressive (e.g., “grabbing utensil” precedes “eating”). (4) Afordance and commonsense constraints. Interactions must respect object afordances (e.g., “stirring with a fork/spoon,” not “stirring with a phone”). A caption may cover multiple actions, but each action span must be temporally localizable. Note that in CUHK-X, captions prioritize action/scene semantics over appearance details that are modality-incongruent. This human verification serves as a reliability gate, yielding captions that are (i) physically plausible within the recorded environments, (ii) temporally and causally coherent, and (iii) aligned with the action ontology.

## 4.3 Put All Things Together

4.3.1 Hardware and Environment Setup. In this section, we describe our hardware and environment configurations. As shown in Fig. 5a, firstly, we use a Goermicro Vzense NYX 650 camera to capture RGB, depth, and infrared data. Next, we use a Texas Instruments IWR6843ISK mmWave radar operating. In addition, we use a Hikvision TB4117 thermal imaging camera for precise temperature measurement. Moreover, we use a TSRV-Q9 AI Tracking Gimbal which is designed for precise automatic tracking and stabilization. In practice, we fix the sensor’s angle and position during data collection. Lastly, we use the Bluetooth 5.0-enabled WitMotion WT9011DCL-BT50 as our 9-axis Inertial Measurement Unit (IMU) for precise tracking of acceleration, angular velocity, and magnetic field. Each participant was equipped with 5 of these devices, with sensors placed on the wrists, ankles, and waist using adjustable bands, shown in Fig. 5b. We collected data from two indoor environments, with a focus on four common room settings: the living room, kitchen, bedroom, and bathroom. Our environmental setup not only enables fine-grained monitoring of human activities but also supports the integration and analysis of data across multiple modalities.<sup>2</sup>

4.3.2 Demographic Characteristics of Participants. We recruited 30 participants (40% male, 60% female) with an age range of20-23 years. BMI ranged from 16.41 to 29.02, with a mean of24.54. Additionally, we collected data on participants’ activity habits, indicating an average session duration of approximately 22.67 minutes and an average exercise frequency of 1.7 times per week, where low, moderate, and high intensities are assigned scores of 1, 2, and 3, respectively. These metrics suggest a relatively balanced distribution of height and weight among participants and highlight their tendency toward low-frequency, short-duration exercise routines. This dataset provides a meaningful baseline for the development and evaluation of computational models or systems aimed at activity recognition and health monitoring, ensuring both generalizability and reliability in human-centric data.

4.3.3 Data Synchronization andAnnotation. To ensure precise alignment across all modalities, we adopt the global time from the host computer as the reference for synchronization. We use a marker, i.e., a director’s board, to define the start and end points of the alignment process, enabling consistent temporal boundaries for all recorded data. RGB data serves as the primary modality for synchronization due to its high temporal resolution and consistency. Radar and IMU data are recorded with timestamps rigorously aligned to the global time, ensuring that all data streams are temporally synchro nized to a high degree of accuracy.

![](images/dcfba33903f592a730adb04e940288a896d70667961690adc8113184ec63b631.jpg)  
(a) Ambient sensors include the Vzense NYX 650 for image sensing, the Texas Instruments IWR6843ISK for radar sensing, and the Hikvision TB4117 for thermal imaging. TSRV-Q9 is an AI tracking gimbal.

![](images/775cf0cb47a6fb022d469013fa73e78993d0fb5c3b57e11862e66e46154d72b1.jpg)  
(b) Wearable sensor includes 5 IMU sensor placements on the wrists, ankles, and waist using bandings, and the WitMotion WT 9011DCL-BT50 IMU module.  
Figure 5: Photos of our ambient and wearable sensor hardware.

For caption data annotation, captions are pre-generated (refer to §4 for more details), and subsequently used during data collection. This process ensures that the descriptions of each video segment are naturally aligned with the corre sponding actions, minimizing annotation errors. In addition to caption-level alignment, individual action annotations are performed with meticulous care. Each video segment is manually labeled and segmented on a frame-by-frame basis to achieve the highest possible precision. Special attention is given to segment transitions and ambiguous actions to avoid misalignment or mislabeling, which can significantly impact downstream tasks. This manual process provides an accurate foundation for training and evaluating computational models. We provide more details of data statistics and data visualizations in §B.1 and §B.2, respectively.

## 5 EXPERIMENTAL SETUP

Here, we describe our tasks, baseline, metrics, and implementation details. Note that in this paper, LLMs are used for generality without distinguishing between modalities.

## 5.1 Tasks Descriptions

5.1.1 HAR Task. HAR is a task focused on identifying and classifying human activities such as walking, running, sitting, and standing from sensor data. We define 40 classes across seven categories for recognition as our HAR tasks.

5.1.2 HAU Tasks. HAU goes beyond basic HAR by capturing richer semantic information. Unlike HAR, which focuses on predefined actions, HAU seeks to understand the context of action sequences, including spatiotemporal semantics, relationships between actions, their order, and interactions with objects or the environment. In particular, we define four subtasks in HAU as follows:

• Caption Comparison: This sub-task involves comparing the captions generated by the LLMs with the ground truth captions to evaluate the LLMs’ capability for accurate description generation.

• Context Analysis: This task requires that the LLMs must identify the correct context exhibited by the participants. In particular, we hope LLMs can recognize when actions are performed in a relaxed, calm, or hurried manner.

• Sequential Action Reordering: The model observes data containing actions in a shufled order and accurately reorders them into the correct sequence.

• Action Selection: The LLMs observe the data to select the correct actions from a predefined pool of 40 actions.

5.1.3 HARn Task. HARn goes beyond HAU’s semantic understanding by adding reasoning capabilities to infer intentions, causal relationships, and logical action sequences, which involves predicting outcomes. Specifically, the model must predict the next action from a provided list based on a series of preceding actions.

## 5.2 Baseline and Metrics

5.2.1 HAR Task. We use ResNet-50 [18] for its visual recognition efectiveness. Radar data is processed with PointNet [37], enhanced by feature engineering to capture spatial characteristics. Skeleton data employs MotionBert [66], using a dualstream transformer and a multilayer perceptron, with 17 3D joints extracted via Human3.6M-compliant pose estimation models [12]. IMU data is handled by a 1D-CNN [48] with three convolutional layers and transformer encoder layers [51] fol lowed by a linear classification head. HAR task evaluation uses Accuracy, Precision, Recall, and F1-score.

5.2.2 HAU and HARn Tasks. We evaluate these two tasks via the latest video LLMs and visual reasoning LLMs, including InternVL2.5-2B (InternVL-2B) [10], InternVL2.5-8B (InternVL-8B) [10], QwenVL2.5-3B/7B (QwenVL-3B/7B) [3], VideoLLaVA-7B (VLLaVA-7B) [28], and VChatR1-7B [27]. For the caption comparison task in HAU, we use metrics including BERTScore (F1-Score), ROUGE-1, ROUGE-L, BLEU-1, and METEOR scores. Additionally, we use accuracy to evaluate the emotion analysis, sequential action reordering, activity selection, and HARn task.

## 5.3 Implementation Details

For both training schemes, we set the learning rate to 0.001, use a batch size of 64, and update parameters with the Adam [24] optimizer. For HAR tasks, we randomly split 80% as the training set and 20% as the testing set in all modalities. For RGB, IR, thermal, and depth modalities, models are initialized with ResNet-50 pre-trained weights, while for IMU and radar modalities, baseline models are initialized using Kaiming initialization [18] since no general pre-trained models are available for these modalities. Besides, for the skeleton modality, we initialize MotionBERT [66] as our backbone with weights pre-trained on the NTU RGB+D dataset [42]. We provide the process details of depth data in §A.5.

All video LLMs are evaluated under a zero-shot paradigm using their default configurations and task-specific system prompts.We directly use the whole video clip as our input. In HAU tasks, models receive diferent prompts: (1) captioning-“Describe what the person in the video is doing. You can briefly mention the background or setting, but focus mainly on understanding the person’s actions.”; (2) emotion analysis-“What emotion does the person experience while performing the activities?”; (3) sequential action reordering-“What activity is the person performing in the video? You must choose only from the following activities: {Class Set}. You can choose multiple activities if necessary.”; (4) action selection-“Please sort the following activity lists in chronological order based on the video content.”.; (5) HARn-“What activity is the person likely to do next?”.

## 6 BENCHMARKS

We present three benchmarks in CUHK-X: HAR, HAU, and HARn. For HAU and HARn we report results for vision only. The main reason is that we have strong and widely used LLMs exist for RGB, for example QwenVL, while comparable models for sensors such as IMU or mmWave are not yet available. Note that a direct comparison across these very diferent modalities would not be fair. Our results are meant to show that each

Table 3: Overall Performance of HAR Task in CUHK-X.

<table><tr><td>Modality</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>RGB</td><td>90.89%</td><td>92.24%</td><td>91.02%</td><td>91.28%</td></tr><tr><td>Depth</td><td>90.46%</td><td>91.76%</td><td>90.75%</td><td>90.93%</td></tr><tr><td>IR</td><td>90.22%</td><td>91.53%</td><td>89.94%</td><td>90.46%</td></tr><tr><td>Thermal</td><td>92.57%</td><td>93.54%</td><td>93.50%</td><td>93.36%</td></tr><tr><td>mmWave</td><td>46.63%</td><td>48.29%</td><td>46.63%</td><td>44.53%</td></tr><tr><td>IMU</td><td>45.52%</td><td>40.84%</td><td>38.00%</td><td>38.32%</td></tr><tr><td>Skeleton</td><td>79.08%</td><td>91.46%</td><td>79.08%</td><td>84.17%</td></tr></table>

modality contains useful task relevant information, not to rank the modalities or claim that one is the best.

## 6.1 Benchmark of HAR

To verify that the CUHK-X dataset contains suficiently new knowledge for the HAR tasks in diferent data modalities, we provide a benchmark for the HAR task. The remaining content is structured around addressing the following two questions: (1) Does the dataset contain valuable knowledge? (2) What are the challenges in this task?

6.1.1 OverallPerformance. As shown in Table 3, performance varies by modality, with an overall accuracy of 76.52%. With standard supervised training, vision-based inputs work best. Thermal gives the top results, with accuracy 92.57% and F1 score 93.36%. Depth and RGB follow, with F1 scores of 90.93% and 91.28%. IR is lower than the other visual channels but still strong, with F1 90.46%. Skeleton features also perform well, with accuracy 79.08% and F1 84.17%. In contrast, other sensors are much lower: mmWave reaches 46.63% accuracy and 44.53% F1, and IMU reaches 45.52% accuracy and 38.32% F1. These gaps likely come from the lower spatial resolution and signal-to-noise ratio of mmWave, the sensitivity of IMU to placement and orientation, and larger shifts across trials for these sensors. Overall, the results show that the dataset provides useful information for HAR.

6.1.2 Long-tailed Class Performance. As shown in Fig.13, CUHK-X exhibits a long-tailed class-frequency distribution, which may afect the results. The imbalance ratio is approximately 10 (i.e., the most frequent class appears about ten times as often as the rarest), indicating a moderate, though not extreme, level of imbalance [45]. This issue can be alleviated with standard techniques such as data resampling [60], data augmentation[20, 21], or balanced-loss objectives [45]. In particular, as shown in Fig. 6a, applying class-balanced resampling on RGB, IMU and Skeleton modalities in CUHK-X yields a measurable improvement in accuracy. In particular, resampling consistently improves accuracy across modalities, with the largest gain for RGB from 90.89% to 96.16%, and smaller but noticeable gains for IMU.

6.1.3 Cross-subject Performance. We evaluate cross-subject performance ofCUHK-X using a leave-one-subject-out (LOSO)

![](images/44265c76d37720577feb8bbc6c29ef551d7ed53a826182fc7186b82d7929d42b.jpg)

![](images/90d3cce689edf4a47d919fe358e324d9bbb4d5a6a48d72a614cf7827c58411e1.jpg)  
(b) Cross-subject.  
Figure 6: Long-tail (RGB, IMU, Skeleton) and crosssubject (RGB) performance. w/o LT means long-tail classes are removed; Contra. means contrastive learning is added; w/o CD means cross-domain data are excluded.

protocol. In each fold, one subject is held out for testing and the remaining subjects are used for training; results are averaged over five folds. As shown in Fig. 6b, the Baseline (LOSO on RGB only) exhibits a substantial accuracy drop due to subject shift and the long-tailed label distribution. Performance improves as we progressively mitigate these factors: removing long-tailed classes (w/o LT) yields a clear gain; adding contrastive learning (Contra.) further strengthens subject invariant representations; and excluding cross-domain data (w/o CD) achieves the best result by eliminating domain shift, i.e., reaching 56.38%. Note that in CUHK-X, CD denotes that we use training and testing data in the same physical environment. Compared with the performance of conventional HAR, accuracy drops markedly in the cross-domain setting, which is intrinsically challenging due to domain shift; even state-ofthe-art methods report only ∼ 60% accuracy in this setup [30].

## 6.2 Benchmark of HAU

In HAU benchmark, our goal is to benchmark the task per formance of diferent models and diferent modalities of the following four sub-tasks.

6.2.1 Results of Caption Comparison. As shown in Table 4, diferent models excel on diferent metrics and modalities. VLLaVA-7B achieves the best BERTScore F1 and the highest ROUGE-1/ROUGE-L on the depth and thermal modalities. QwenVL-3B attains the top ROUGE and BLEU-1 scores on RGB and IR and remains competitive elsewhere. VR1Chat-7B frequently yields the best BLEU-1 and METEOR (e.g., RGB and IR), indicating strong fluency. In contrast, InternVL-2B/8B obtain decent BERTScore F1 but lag behind on ROUGE and BLEU. Overall, higher-capacity models (e.g., 7B) tend to outperform the InternVL-2B/8B baselines, although the 3B Qwen model is a notable exception that matches or surpasses some 7B models on several metrics.

6.2.2 Results ofContext Analysis. As shown in Fig. 7a, the average accuracy is 50.52%. In particular, VLLaVA-7B delivers the best overall context accuracy across modalities, leading on RGB, IR, and Depth. QwenVL-3B follows closely and remains competitive on Thermal. InternVL-2B and InternVL-8B stay at the lower end, with accuracies roughly between 24 and 35 percent across modalities, suggesting that although these architectures work well for general captioning, they are weaker at modeling contextual cues. VR1Chat-7B achieves mid-range results, typically around 42 to 50 percent, with slightly better scores on RGB and Depth than on IR and Thermal. The spread between models is largest on IR and Depth, where the best systems approach 80 percent while the weakest remain below 30 percent, underscoring the challenge of interpreting non-RGB signals.

Table 4: Results of caption comparison. We report BERTScore F1 (B.-F1), ROUGE-1 (R.-1), ROUGE-L (R.-L), BLEU-1 (B.-1), and METEOR (MET.).

<table><tr><td>Model</td><td>B.-F1</td><td>R.-1</td><td>R.-L</td><td>B.-1</td><td>MET.</td></tr><tr><td colspan="6">RGB</td></tr><tr><td>InternVL-2B</td><td>84.39%</td><td>4.33%</td><td>3.64%</td><td>0.61%</td><td>3.97%</td></tr><tr><td>InternVL-8B</td><td>84.07%</td><td>3.04%</td><td>2.53%</td><td>0.72%</td><td>3.63%</td></tr><tr><td>QwenVL-3B</td><td>86.22%</td><td>18.40%</td><td>13.80%</td><td>21.46%</td><td>19.89%</td></tr><tr><td>QwenVL-7B</td><td>85.47%</td><td>14.79%</td><td>12.05%</td><td>18.04%</td><td>22.21%</td></tr><tr><td>VLLaVA-7B</td><td>86.40%</td><td>16.12%</td><td>12.77%</td><td>12.86%</td><td>11.58%</td></tr><tr><td>VR1Chat-7B</td><td>86.24%</td><td>17.42%</td><td>13.66%</td><td>21.62%</td><td>23.18%</td></tr><tr><td colspan="6">Depth</td></tr><tr><td>InternVL-2B</td><td>84.09%</td><td>4.63%</td><td>3.95%</td><td>0.53%</td><td>3.67%</td></tr><tr><td>InternVL-8B</td><td>83.95%</td><td>2.89%</td><td>2.35%</td><td>0.73%</td><td>3.52%</td></tr><tr><td>QwenVL-3B</td><td>85.03%</td><td>15.00%</td><td>10.76%</td><td>18.58%</td><td>16.89%</td></tr><tr><td>QwenVL-7B</td><td>84.55%</td><td>12.70%</td><td>10.65%</td><td>16.64%</td><td>18.98%</td></tr><tr><td>VLLaVA-7B</td><td>85.94%</td><td>16.31%</td><td>13.53%</td><td>12.07%</td><td>10.37%</td></tr><tr><td>VR1Chat-7B</td><td>84.69%</td><td>14.17%</td><td>11.49%</td><td>17.73%</td><td>19.39%</td></tr><tr><td colspan="6">IR</td></tr><tr><td>InternVL-2B</td><td>84.24%</td><td>4.38%</td><td>3.68%</td><td>0.58%</td><td>4.09%</td></tr><tr><td>InternVL-8B</td><td>84.22%</td><td>3.05%</td><td>2.54%</td><td>0.64%</td><td>3.68%</td></tr><tr><td>QwenVL-3B</td><td>86.49%</td><td>18.56%</td><td>13.99%</td><td>22.17%</td><td>18.85%</td></tr><tr><td>QwenVL-7B</td><td>85.38%</td><td>14.71%</td><td>11.93%</td><td>18.25%</td><td>21.50%</td></tr><tr><td>VLLaVA-7B</td><td>86.25%</td><td>15.64%</td><td>12.81%</td><td>12.25%</td><td>10.78%</td></tr><tr><td>VR1Chat-7B</td><td>86.03%</td><td>16.51%</td><td>12.96%</td><td>21.23%</td><td>22.05%</td></tr><tr><td colspan="6">Thermal</td></tr><tr><td>InternVL-2B</td><td>84.30%</td><td>5.05%</td><td>4.28%</td><td>0.38%</td><td>3.93%</td></tr><tr><td>InternVL-8B</td><td>83.85%</td><td>2.74%</td><td>2.24%</td><td>0.73%</td><td>3.37%</td></tr><tr><td>QwenVL-3B</td><td>85.04%</td><td>14.78%</td><td>11.40%</td><td>18.13%</td><td>16.95%</td></tr><tr><td>QwenVL-7B</td><td>84.48%</td><td>12.44%</td><td>10.31%</td><td>15.41%</td><td>20.24%</td></tr><tr><td>VLLaVA-7B</td><td>85.85%</td><td>16.24%</td><td>13.32%</td><td>11.31%</td><td>10.16%</td></tr><tr><td>VR1Chat-7B</td><td>84.94%</td><td>14.23%</td><td>11.51%</td><td>17.90%</td><td>19.86%</td></tr></table>

6.2.3 Results ofSequential Action Reordering. As shown in Fig. 7b, the average accuracy is 47.24% and no single model dominates across all modalities. We notice that InternVL-8B achieves the best accuracy on RGB and IR, reaching roughly three quarters in both cases. QwenVL-3B edges out the others on Depth, with scores in the mid-sixties. VR1Chat-7B is strongest on Thermal, around sixty percent. QwenVL-7B is steady across modalities but does not lead any of them. VLLaVA-7B performs poorly, i.e., close to chance on every modality, while InternVL-2B also sits toward the lower end. Across modalities, IR and RGB are generally the most infor mative for this task, Depth is in the middle, and Thermal is the most challenging. These results suggest that architec ture–modality compatibility matters: capacity helps in places, but the top results come from diferent models on diferent sensors.

![](images/b24411317dd915dc8385d754a75956761ea6882b8aae76f24dd921c56e34f5ab.jpg)

![](images/55ad846fe9da6104b7268beecc1654ff5b3b38fa144475b2c6aacd6cdcf1e3f1.jpg)  
(b) Sequential Action Reordering  
Figure 7: Accuracy of context analysis and sequential action reordering in CUHK-X.

6.2.4 Results ofAction Selection . As shown in Fig. 8, the average accuracy is 24.54%. We notice that QwenVL-7B generally attains the best or near-best scores across all three metrics and modalities, indicating strong action-selection ca pability, VLLava-7B is consistently competitive, with notable strength on infrared and thermal inputs. InternVL-8B delivers mid-range results, with its strongest performance on RGB, while InternVL-2B trails the other models, reflecting the lim itations of a smaller-capacity model. Across modalities, RGB and infrared typically outperform depth and thermal, suggesting that these signals provide more task-relevant information. These findings underscore thejoint importance ofmodel scale and sensing modality for robust action selection.

## 6.3 Benchmark ofHARn

We have evaluated the performance of LVLMs in inferring in tentions and causal relationships in human action sequences.

6.3.1 Results ofHARn. As shown in Fig. 9, the average accuracy is 70.25%. We notice that VLLava-7B leads on RGB and infrared and remains competitive on depth. QwenVL-7B performs consistently well across all modalities and is close to the top overall. QwenVL-3B stands out on depth, surpassing several larger models. We found that InternVL-8B and VR1Chat-7B deliver mid-range results, while InternVL-2B trails the others. On average, depth and infrared yield higher accuracy than RGB, indicating that these signals carry more in formation for HARn. These findings highlight the combined impact of model capacity and sensing modality on robust human-activity reasoning.

![](images/0ce61ef75309a500d064faac7dd32f765bd076d4efba6094a76cdd79f717ad4a.jpg)  
Figure 8: Accuracy of action selection tasks in HAU.

6.3.2 Whyreasoningmodelworksin HARn? Fig. 10 illustrates the superiority ofreasoning-based models in HARn tasks. Unlike captioning models, e.g., Qwen-7B, InternVL-8B, that misinterpret superficial cues, the reasoning model, i.e., VR1Chat, leverages contextual understanding and logical inference. It associates observed actions, such as interacting with items on the table, with the most likely next action (i.e., “Getting Dressed”). Additionally, the reasoning model excels in handling ambiguity and provides transparent explanations, enhancing interpretability. This capability to integrate temporal reasoning and contextual synthesis makes reasoning models more reliable for HARn tasks, where understanding intent and action progression is critical.

## 7 DISCUSSIONS

Due to the page limitation, we have provided more discussion of CUHK-X in our Appendix §C.

Limitations. CUHK-X is a controlled dataset collected in two indoor environments with 30 participants with ages from 20 to 23. As such, it lacks population-level and ecological diversity, and generalization to other settings, long-horizon activities, or populations with diferent motor patterns, such as children, older adults, the individuals with mobility impairments, is uncertain. Because actions are elicited from scene-level captions, instruction-following may shift motion distributions relative to naturalistic behavior despite preliminary, small-scale evidence of caption similarity. In future releases, we may refine our adjudication protocol, such as reporting inter-rater reliability for the initial independent annotations, which can provide a more comprehensive evaluation.

Figure 9: Results of HARn.  
![](images/c87e63c27c5077165ad67823206821f4e392448d3b4305798dde1b89fb870115.jpg)

FutureDirections. CUHK-X can scale along two axes: (1) adding interaction actions among participants and longer routines; and (2) augmenting the current seven modalities, with complementary signals such as audio, tactile/contact sensing, and lightweight physiology (heart rate, EEG). We also plan to broaden the participant pool and environments to strengthen generalizability. In parallel, CUHK-X serves as a standard benchmark for HAR, and as a testbed for LLM-based action understanding and reasoning. Its tightly synchronized mul timodal streams make it a practical educational resource for teaching sensor fusion, and multimodal reasoning.

## 8 RELATED WORKS

Human Action Recognition Datasets. HAR analyzes and classifies human actions using various sensors. Vision-based datasets mainly utilize RGB and RGB-D data to capture activities. For example, NTU-60 [42] provides 56,880 videos of daily and health-related actions, while UTD [7] records 27 activities from 8 subjects for classification. Similarly, PKU-MMD [11] and NTU-120 [31] leverage RGB, depth, and skeleton data, supporting 66 and 120 actions, respectively. Sensor-based datasets use wearable or environmental sensors like IMUs, gyroscopes, or radar. UMAFall [6] employs IMU sensors on the chest, waist, wrist, and ankle for fall detection, while Epic-Kitchen [13] combines IMU, RGB, and optical flow to analyze over 90,000 action segments in kitchen environments. Smaller datasets, such as USC [65], Shoaib [44], HHAR [46], and UCI [41], focus on common activities like walking using IMU data. Radar-based datasets, such as HuPR [26], integrate radar and RGB for privacy-preserving action recognition. The emerging dataset Thermal-IM [49] employs thermal imaging and multimodal data to address challenges such as lighting variations and occlusion, enabling efective long-term track ing. MM-Fi [58] integrates RGB, depth, and radar, ofering over 320,000 samples for 27 activities conducted by 40 subjects.

![](images/c8db5ec2524c6b7d6c8833cfcbdf1bdc74897412dcc7bb6eb786b1571f66f0ea.jpg)  
Figure 10: An illustration of analysis why the reasoning model performs well than the captioning model.

While limited in scale, mRI [1] also combines IMU, covering 12 activities performed by 20 subjects. However, existing datasets lack comprehensive HAR data from diverse IoT devices.

Human Action Understanding Datasets. HAU involves comprehending actions through perceptual, contextual, and experiential integration, covering recognition, intention, and narrative understanding. Multimodal datasets such as PKU-MMD [11] and Ego-Exo4D [17] support the evaluation of algorithms for understanding complex activities. Captioningbased datasets such as Ego-4D [16] and Ego-Exo4D include 3,670 hours of videos with narrations to enrich activity understanding, while Tarsier2 [63] uses large language models for detailed descriptions. Reasoning-based datasets like ActivityNet-QA [61] and Next-QA [55] focus on spatial, temporal and causal reasoning, with annotated question-answer pairs to enhance deeper video content understanding. DailySTR [39] further leverages the VirtualHome-AIST simulator to create a video-based dataset comprising a total of 80,573 question-answer (QA) pairs. However, these datasets often focus on single modalities or involve high annotation costs. CUHK-X addresses these gaps as the first multimodal dataset for HAU, integrating understanding and reasoning across modalities to advance human action comprehension.

## 9 CONCLUSION

In this paper, we present CUHK-X, a large-scale multimodal dataset and benchmark in HAR, HAU and HARn with ofering 58,445 samples across seven modalities and two environments. With three carefully designed benchmarks encompassing six tasks, our results demonstrate the robustness of CUHK-X in validating state-of-the-art models across three tasks.

## REFERENCES

[1] Sizhe An, Yin Li, and Umit Ogras. 2022. mri: Multi-modal 3d human pose estimation dataset using mmwave, rgb-d, and inertial sensors. Advances in neural information processing systems 35 (2022), 27414–27426.

[2] Godfred Anakpo, Zanele Nqwayibana, and Syden Mishi. 2023. The impact of work-from-home on employee performance and productivity: a systematic review. Sustainability 15, 5 (2023), 4529.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923 (2025).

[4] Daumantas Bočkus, Timo Tammi, Elli Vento, and Raija Komppula. 2023. Wellness tourism service preferences and their linkages to motivational factors: a multiple case study. International Journal ofSpa and Wellness 6, 1 (2023), 78–108.

[5] Fabian Caba Heilbron, Victor Escorcia, Bernard Ghanem, and Juan Carlos Niebles. 2015. Activitynet: A large-scale video benchmark for human activity understanding. In Proceedings ofthe ieee conference on computer vision and pattern recognition. 961–970.

[6] Eduardo Casilari, Jose A Santoyo-Ramón, and Jose M Cano-García. 2017. Umafall: A multisensor dataset for the research on automatic fall detection. Procedia Computer Science 110 (2017), 32–39.

[7] Chen Chen, Roozbeh Jafari, and Nasser Kehtarnavaz. 2015. UTD MHAD: A multimodal dataset for human action recognition utilizing a depth camera and a wearable inertial sensor. In 2015 IEEE International conference on image processing (ICIP). IEEE, 168–172.

[8] Hongkai Chen, Sirajum Munir, and Shan Lin. 2022. RFCam: Uncertainty aware fusion of camera and wi-fi for real-time human identification with mobile devices. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 6, 2 (2022), 1–29.

[9] Richard Chen, Filip Jankovic, Nikki Marinsek, Luca Foschini, Lampros Kourtis, Alessio Signorini, Melissa Pugh, Jie Shen, Roy Yaari, Vera Maljkovic, et al. 2019. Developing measures of cognitive impairment in the real world from consumer-grade multimodal sensor streams. In Proceedings ofthe 25th ACM SIGKDD international conference on knowledge discovery & data mining. 2145–2155.

[10] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. 2024. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 24185–24198.

[11] Liu Chunhui, Hu Yueyu, Li Yanghao, Song Sijie, and Liu Jiaying. 2017. PKU-MMD: A Large Scale Benchmark for Continuous Multi-Modal Human Action Understanding. arXiv preprint arXiv:1703.07475 (2017).

[12] MMPose Contributors. 2020. OpenMMLab Pose Estimation Toolbox and Benchmark. https://github.com/open-mmlab/mmpose.

[13] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Evangelos Kazakos, Jian Ma, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. 2022. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal ofComputer Vision (2022), 1–23.

[14] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. Ieee, 248–255.

[15] Leah R Gerber, Zachary Reeves-Blurton, Nika Gueci, Gwenllian D Iacona, JA Beaudette, and Teri Pipe. 2023. Practicing mindfulness in addressing the biodiversity crisis. Conservation Science and Practice 5, 7 (2023), e12945.

[16] Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. 2022. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 18995–19012.

[17] Kristen Grauman, Andrew Westbury, Lorenzo Torresani, Kris Kitani, Jitendra Malik, Triantafyllos Afouras, Kumar Ashutosh, Vijay Baiyya, Siddhant Bansal, Bikram Boote, et al. 2024. Ego-exo4d: Understanding skilled human activity from first-and third-person perspectives. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 19383–19400.

[18] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2015. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision. 1026–1034.

[19] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

[20] Siyang Jiang, Wei Ding, Hsi-Wen Chen, and Ming-Syan Chen. 2022. PGADA: Perturbation-guided adversarial alignment for few-shot learning under the support-query shift. In Pacific-Asia Conference on Knowledge Discovery and Data Mining. Springer, 3–15.

[21] Siyang Jiang, Rui Fang, Hsi-Wen Chen, Wei Ding, and Ming-Syan Chen. 2023. Dual adversarial alignment for realistic support-query shift few-shot learning. arXiv preprint arXiv:2309.02088 (2023).

[22] Siyang Jiang, Xian Shuai, and Guoliang Xing. 2024. ArtFL: Exploiting data resolution in federated learning for dynamic runtime inference via multi-scale training. In 2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN). IEEE, 27–38.

[23] Siyang Jiang, Bufang Yang, Lilin Xu, Mu Yuan, Yeerzhati Abudunuer, Kaiwei Liu, Liekang Zeng, Hongkai Chen, Zhenyu Yan, Xiaofan Jiang, et al. 2025. An LLM-Empowered Low-Resolution Vision System for On-Device Human Behavior Understanding. arXiv preprint arXiv:2505.01743 (2025).

[24] Diederik P Kingma and Jimmy Ba. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[25] Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, et al. 2020. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. Internationaljournalofcomputervision 128, 7 (2020), 1956–1981.

[26] Shih-Po Lee, Niraj Prakash Kini, Wen-Hsiao Peng, Ching-Wen Ma, and Jenq-Neng Hwang. 2023. HuPR: A Benchmark for Human Pose Estimation Using Millimeter Wave Radar. In Proceedings ofthe IEEE/CVFWinter Conference on Applications ofComputer Vision (WACV). 5715–5724.

[27] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2023. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355 (2023).

[28] Bin Lin, Bin Zhu, Yang Ye, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-llava: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122 (2023).

[29] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024).

[30] Hanchao Liu, Yujiang Li, Tai-Jiang Mu, and Shi-Min Hu. 2024. Recovering complete actions for cross-dataset skeleton action recognition. Advances in Neural Information Processing Systems 37 (2024), 92055–92081.

[31] Jun Liu, Amir Shahroudy, Mauricio Perez, Gang Wang, Ling-Yu Duan, and Alex C Kot. 2019. Ntu rgb+ d 120: A large-scale benchmark for 3d human activity understanding. IEEE transactions on pattern analysis

and machine intelligence 42, 10 (2019), 2684–2701.

[32] Michelle E Mlinac and Michelle C Feng. 2016. Assessment of activities of daily living, self-care, and independence. Archives of Clinical Neuropsychology 31, 6 (2016), 506–516.

[33] Debjyoti Mondal, Suraj Modi, Subhadarshi Panda, Rituraj Singh, and Godawari Sudhakar Rao. 2024. Kam-cot: Knowledge augmented multimodal chain-of-thoughts reasoning. In Proceedings of the AAAI conference on artificial intelligence, Vol. 38. 18798–18806.

[34] U.S. Department of Labor. 2013. American Time Use Survey. http://www.bls.gov/tus/ Accessed: 2025-04-23.

[35] Xiaomin Ouyang, Xian Shuai, Yang Li, Li Pan, Xifan Zhang, Heming Fu, Sitong Cheng, Xinyan Wang, Shihua Cao, Jiang Xin, et al. 2024. AD Marker: A Multi-Modal Federated Learning System for Monitoring Dig ital Biomarkers ofAlzheimer’s Disease. In Proceedings ofthe 30th Annual International Conference on Mobile Computing and Networking. 404–419.

[36] Xiaomin Ouyang, Xian Shuai, Jiayu Zhou, Ivy Wang Shi, Zhiyuan Xie, Guoliang Xing, and Jianwei Huang. 2022. Cosmo: contrastive fusion learning with small data for multimodal human activity recognition. In Proceedings of the 28th Annual International Conference on Mobile Computing And Networking. 324–337.

[37] Charles R Qi, Hao Su, Kaichun Mo, and Leonidas J Guibas. 2017. Pointnet: Deep learning on point sets for 3d classification and segmentation. In Proceedings ofthe IEEE conference on computer vision and pattern recognition. 652–660.

[38] Wenhao Qi, Xiaohong Zhu, Bin Wang, Yankai Shi, Chaoqun Dong, Shiying Shen,Jiaqi Li, Kun Zhang, Yunfan He, Mengjiao Zhao, et al. 2025. Alzheimer’s disease digital biomarkers multidimensional landscape and AI model scoping review. npj Digital Medicine 8, 1 (2025), 366.

[39] Yue Qiu, Shusaku Egami, Ken Fukuda, Natsuki Miyata, Takuma Yagi, Kensho Hara, Kenji Iwata, and Ryusuke Sagawa. 2024. DailySTR: A Daily Human Activity Pattern Recognition Dataset for Spatio-temporal Reasoning. In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 357–363.

[40] TJ Quinn, K McArthur, G Ellis, and DJ Stott. 2011. Functional assessment in older people. Bmj 343 (2011).

[41] Jorge-L Reyes-Ortiz, Luca Oneto, Albert Samà, Xavier Parra, and Davide Anguita. 2016. Transition-aware human activity recognition using smartphones. Neurocomputing 171 (2016), 754–767.

[42] Amir Shahroudy, Jun Liu, Tian-Tsong Ng, and Gang Wang. 2016. Ntu rgb+ d: A large scale dataset for 3d human activity analysis. In Proceedings of the IEEE conference on computer vision and pattern recognition. 1010–1019.

[43] SungUk Shin and Youngjoon Kim. 2025. Enhancing Graph Of Thought: Enhancing Prompts with LLM Rationales and Dynamic Temperature Control. In The Thirteenth International Conference on Learning Representations.

[44] Muhammad Shoaib, Stephan Bosch, Ozlem Durmaz Incel, Hans Scholten, and PaulJM Havinga. 2014. Fusion ofsmartphone motion sen sors for physical activity recognition. Sensors 14, 6 (2014), 10146–10176.

[45] Xian Shuai, Yulin Shen, Siyang Jiang, Zhihe Zhao, Zhenyu Yan, and Guoliang Xing. 2022. BalanceFL: Addressing class imbalance in long-tail federated learning. In IPSN. IEEE, 271–284.

[46] Allan Stisen, Henrik Blunck, Sourav Bhattacharya, Thor Siiger Prentow, Mikkel Baun Kjærgaard, Anind Dey, Tobias Sonne, and Mads Møller Jensen. 2015. Smart devices are diferent: Assessing and mitigatingmo bile sensing heterogeneities for activity recognition. In Proceedings of the 13th ACM conference on embedded networked sensor systems. 127–140.

[47] Yue-meng Sun, Zhi-yun Wang, Yuan-yuan Liang, Chen-wei Hao, and Chang-he Shi. 2024. Digital biomarkers for precision diagnosis and monitoring in Parkinson’s disease. NPJ digital medicine 7, 1 (2024), 218.

[48] Wensi Tang, Guodong Long, Lu Liu, Tianyi Zhou, Michael Blumenstein, and Jing Jiang. 2020. Omni-scale cnns: a simple and efective kernel

size configuration for time series classification. arXiv preprint arXiv:2002.10061 (2020).

[49] Zitian Tang, Wenjie Ye, Wei-Chiu Ma, and Hang Zhao. 2023. What Happened 3 Seconds Ago? Inferring the Past with Thermal Imaging. In CVPR.

[50] Zitian Tang, Wenjie Ye, Wei-Chiu Ma, and Hang Zhao. 2023. What happened 3 seconds ago? inferring the past with thermal imaging. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 17111–17120.

[51] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems 30 (2017).

[52] Fei Wang, Yizhe Lv, Mengdie Zhu, Han Ding, and Jinsong Han. 2024. Xrf55: A radio frequency dataset for human indoor action analysis. Proceedings ofthe ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 8, 1 (2024), 1–34.

[53] Jiawei Wang, Liping Yuan, Yuchen Zhang, and Haomiao Sun. 2024. Tarsier: Recipes for training and evaluating large video description models. arXiv preprint arXiv:2407.00634 (2024).

[54] Haiyang Wu, Kaiwei Liu, Siyang Jiang, Zhihe Zhao, Zhenyu Yan, and Guoliang Xing. 2024. Demo abstract: Caringfm: An interactive in-home healthcare system empowered by large foundation models. In 2024 23rd ACM/IEEE International Conference on Information Processing in Sensor Networks (IPSN). IEEE, 255–256.

[55] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. 2021. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition. 9777–9786.

[56] Bufang Yang, Yunqi Guo, Lilin Xu, Zhenyu Yan, Hongkai Chen, Guoliang Xing, and Xiaofan Jiang. 2025. Socialmind: Llm-based proactive ar social assistive system with human-like perception for in-situ live interactions. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies 9, 1 (2025), 1–30.

[57] Bufang Yang, Siyang Jiang, Lilin Xu, Kaiwei Liu, Hai Li, Guoliang Xing, Hongkai Chen, Xiaofan Jiang, and Zhenyu Yan. 2024. Drhouse: An llmempowered diagnostic reasoning system through harnessing outcomes from sensor data and expert knowledge. Proceedings of the ACM on Interactive, Mobile, Wearable andUbiquitous Technologies 8, 4 (2024), 1–29.

[58] Jianfei Yang, He Huang, Yunjiao Zhou, Xinyan Chen, Yuecong Xu, Shenghai Yuan, Han Zou, Chris Xiaoxuan Lu, and Lihua Xie. 2023. Mm-fi: Multi-modal non-intrusive 4d human dataset for versatile wireless sensing. Advances in Neural Information Processing Systems 36 (2023), 18756–18768.

[59] Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Tom Grifiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems 36 (2023), 11809–11822.

[60] Rongguang Ye, Yantong Guo, Xian Shuai, Rongye Ye, Siyang Jiang, and Hui Jiang. 2023. Licam: Long-tailed instance segmentation with real-time classification accuracy monitoring. Journal of Circuits, Systems and Computers 32, 02 (2023), 2350032.

[61] Zhou Yu, Dejing Xu, Jun Yu, Ting Yu, Zhou Zhao, Yueting Zhuang, and Dacheng Tao. 2019. Activitynet-qa: A dataset for understanding complex web videos via question answering. In AAAI, Vol. 33. 9127–9134.

[62] Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. 2025. Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding. arXiv preprint arXiv:2501.07888 (2025).

[63] Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, and Yuan Lin. 2025. Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding.

arXiv:2501.07888 [cs.CV] https://arxiv.org/abs/2501.07888

[64] Liyu Zhang, Yizhen Wang, Wenjie Du, Kwun Ho Liu, and Xiaomin Ouyang. 2025. Demo Abstract: An LLM-Powered Multimodal Mobile Sensing System for Personalized and Interactive Health Behavior Analysis. In SenSys. 720–721.

[65] Mi Zhang and Alexander A Sawchuk. 2012. USC-HAD: A daily activity dataset for ubiquitous activity recognition using wearable sensors. In Proceedings of the 2012 ACM conference on ubiquitous computing. 1036–1043.

[66] Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, and Yizhou Wang. 2023. Motionbert: A unified perspective on learning human motion representations. In CVPR. 15085–15099.

## APPENDIX

## A OVERVIEW AND DETAILS

## A.1 Overview of CUHK-X

We introduce CUHK-X, a large-scale multimodal dataset designed to advance human action recognition (HAR), understanding(HAU),andnext-actionreasoning(HARn).Asshown on the left of the figure, we first build a Scene-based Caption Generation framework to avoid spatiotemporal inconsistency and provide precise ground truth. Guided by ATUS, we define seven action themes and select 40 representative actions with reference to HHAR, UCI, and Cosmo. Large language models then compose coherent captions that stitch these actions into everyday scenes—living room, kitchen, bedroom, and bath room—augmented with emotional styles such as relaxed or hurried. Using these captions as ground truth (center), 30 participants enact the scenes in two indoor environments, yield ing over 58,445 samples across seven synchronized modalities: RGB, depth, thermal, infrared, skeleton, IMU, and mmWave. The setup includes a Vzense NYX 650 (depth), TI IWR6843ISK (mmWave), a Hikvision TB4117 (thermal), and five WitMotion WT9011DCL-BT50 IMUs. Each recording is paired with its caption, forming rich data–caption pairs. On the right, CUHK-X supports three benchmarks implemented as eight tasks. HAR performs single-label classification. HAU assesses caption comparison with ground truth, context analysis (e.g., quickly, smoothly, calmly), sequential action reordering to test temporal reasoning, and action selection from a predefined set. HARn predicts the next action from textual descrip tions. We evaluate across all modalities using SOTA mod els, InternVL2.5-2B/8B, QwenVL2.5-3B/7B (captioning) and VideoLLaVA-7B, VideoChatR1-7B (reasoning), and analyze long-tail and cross-subject efects.

## A.2 Motivation Details

As shown in Fig. 2, we provide an illustration of the limitations in current LLMs, such as Tarsier [53] and Tarsier2 [62], which face challenges in achieving accurate HAU with depth and thermal modalities, while RGB performs accurate results. We observed that the SOTA captioning models often make the following mistakes: providing inaccurate action descriptions, missing actions, or sometimes both, as shown in Fig. 2. The main reason is that these models are not trained on this modality or designed for the HAU task. Thus, it is necessary to provide datasets with multimodal synchronized “⟨data, caption⟩” pairs to enable models to understand such information efectively.

## A.3 Hardware Details

A.3.1 Ambient sensors setup. As shown in Fig. 5a, firstly, we use a Goermicro Vzense NYX 650 camera to capture RGB, depth, and infrared data. Specifically, the Vzense NYX 650 cameras ofer a $7 0 ^ { \circ }$ horizontal and 50° vertical field of view, operating at a frame rate of 10 frames per second. Leveraging 940nm infrared light, these cameras are well-suited for both indoor and outdoor environments, even under lowlight or no-light conditions. Next, we use a Texas Instruments IWR6843ISK mmWave radar operating in the 60–64 GHz band. We configure it with a 20 fps frame rate, 0.044 m range resolution, a 5.03 m maximum unambiguous range, a 1.0 m/s maximum radial velocity, and a 0.13 m/s radial velocity resolution. This sensor excels in detecting objects, measuring distances, and tracking motion with high precision. In addition, we use a Hikvision TB4117 thermal imaging camera for precise temperature measurement. Featuring a 120×160- pixel resolution and compact 70×46×22.75 mm dimensions, this device measures temperatures from $3 0 ^ { \circ } \mathrm { C }$ to $4 5 ^ { \circ } \mathrm { C }$ with 25 fps, making it ideal for thermal monitoring. Lastly, we use a TSRV-Q9 AI Tracking Gimbal, a compact (60 × 70 × 185 mm) and lightweight (220 g) device designed for precise automatic tracking and stabilization. Powered by a 3.7V/1200mAh battery, it supports 3.5 hours of continuous tracking. Compatible with devices up to 12 mm thick, it features 360° horizontal rotation and 180° manual vertical adjustment, making it ideal for dynamic content creation. In practice, we fix the sensor’s angle and position during data collection.

A.3.2 Wearable sensors setup. We use the Bluetooth 5.0- enabled WitMotion WT9011DCL-BT50 as our Inertial Measurement Unit (IMU) for precise tracking of acceleration (±16�), angular velocity $( \pm 2 0 0 0 ^ { \circ } / s )$ , and magnetic field (±2 Gauss). It supports output frequencies ranging from 0.2 Hz to 200 Hz and provides angular measurements ofup to $\pm 1 8 0 ^ { \circ }$ (X/Z) and $\pm 9 0 ^ { \circ } \left( \mathrm { Y } \right)$ . Powered by a 130 mAh battery, it delivers up to 40 hours of continuous operation with a maximum transmission range of 50 m in open space with 10 samples per second. Measuring32.5 × 23.5 × 11.6 mm in size and weighing just ${ 9 } \mathrm { g } ,$ each participant was equipped with 5 of these devices, with sensors placed on the wrists, ankles, and waist using adjustable bands, shown in Fig. 5b.

## A.4 Environmental Details

We collected data from two indoor environments, with a focus on four common room settings: the living room, kitchen, bedroom, and bathroom. These rooms were selected to represent a diverse range of daily activities for studying and analyzing human behavior in realistic, everyday scenarios. To ensure systematic data collection, we documented all sensor locations in each room by marking their positions and taking photographic records. This meticulous annotation method provides thorough coverage of typical activities occurring within these spaces, facilitating robust data analysis. The floor plans, as depicted in Fig. 12, ofer detailed spatial representa tions of the two indoor environments, with icons highlighting the exact locations of the sensors deployed for data collection. Specifically, ambient sensors were strategically positioned to optimize coverage and data reliability. Furthermore, each photo depicts a room and serves to visually contextualize the sensor data, providing a visual context to the collected data. Our environmental setup not only enables fine-grained mon itoring of human activities but also supports the integration and analysis of data across multiple modalities.

## A.5 Processing Details of Depth Data.

We observed that directly using this raw data fails to efectively capture the spatial information of depth since the raw depth data is 16-bit. To address this, we process the raw depth data via the two types ofhouse floor plans, shown in Figure 12. In particular, we filtered depth values outside a defined range, replacing them with zero to focus on relevant depth information for each specific environment. In Room 1 of Fig. 12, the depth ranges are set as follows: Living Room [500, 5000], Kitchen [500, 3300], Bedroom [500, 3200], and Bathroom [500, 2800]. In Room 2 of Fig. 12, the depth ranges are slightly diferent: Living Room [500, 4700], Kitchen [500, 3260], Bedroom [500, 3500], and Bathroom [500, 2000]. These ranges are tailored to accommodate the spatial characteristics of each scenario and room.

## B DATA DESCRIPTION

## B.1 Data Statistics

Here, we provide a statistical description of our dataset. As shown in Fig. 13, we show a clear frequency imbalance across human actions. High-frequency actions such as walking, eat ing, sitting down, and drinking water dominate the dataset, with occurrences exceeding 200, reflecting their ubiquity in daily life. Moderately frequent actions, including pouring a drink, stirring utensils, checking time, and standing up, appear between 50 and 150 times, indicating their importance in routine behaviors while being less universal. In contrast, actions such as folding clothes, watching TV, and playing a game are sparsely represented, with fewer than 20 occurrences, likely due to their specific or context-dependent nature. The dataset follows a long-tail distribution, where a small number of actions account for a large proportion of occurrences, while the majority are infrequent. This imbalance is a common characteristic of real-world datasets, which naturally prioritize capturing frequent, everyday behaviors. Despite this, the dataset spans a diverse range of categories, including basic daily activities, work-related tasks, household chores, and physical exercises, providing a rich foundation for human activity recognition. In particular, in CUHK-X, each participant contributes over 30 minutes of footage with more than 100 samples. For example, vision modalities include 4,029 clips, with a total duration of 19 hours and 29 minutes.

## B.2 Data Visualizations

We also provide a visualization ofour multimodal data. Fig. 14 illustrates the multimodal data from both ambient and wearable sensors. We present the visualization of common activities, including sitting, walking, eating, drinking water, and pouring a drink. The data includes RGB, Depth, Thermal, IR, Radar, 3D Skeletons, and IMU signals, representing a comprehensive set ofmodalities for activity recognition. In particular, RGB serves as the primary visual modality, capturing color and spatial context, which is essential for understanding the environment and participants’ actions. Depth data provides spatial structure, highlighting the distance and geometry of objects and participants, while Thermal data visualizes temperature distributions, ofering insights into heat signatures that may not be visible in other modalities. Infrared (IR) enhances visibility in low-light or dark environments, complementing RGB and Thermal data. Radar visualizes motion and spatial dynamics, making it particularly useful for detecting movement patterns. In addition, the 3D Skeletons, extracted using mmpose [12], provide key body joint positions and orientations, enabling precise pose estimation and body movement tracking. These skeletons are overlaid on RGB images for better interpretability of the captured actions. IMU data, collected from five body locations (right arm, left arm, waist, right leg, left leg), includes acceleration, angular velocity, and angles across the X, Y, and Z axes, ofering fine-grained motion analysis. In CUHK-X, each modality not only complements the others but is also capable of functioning independently.

## C DISCUSSION DETAILS

## C.1 Bias Checks of CUHK-X

Because actions in CUHK-X are collected from scene-level captions, instruction-following may shift motion distributions relative to naturalistic behavior. As preliminary evidence, we previously reported (§3.2) a small four-task study indicating high caption similarity. However, caution is warranted when extrapolating to unconstrained, long-horizon activities or to populations with diferent motor patterns (e.g., children, older adults, or individuals with mobility impairments).

![](images/04c52dec50076ab8bbb0d11d516ea596495a748a3b19a3c4aff6af3f45fdcc0a.jpg)

Figure 11: Overview of CUHK-X. We first obtain a set of action categories based on coarse & fine-grained action selection and then generate captions for these actions. We first create scenes based on selected actions and obtain captions. Then, we collect data from 30 participants across seven modalities and obtain the “⟨data, caption⟩” pairs. Lastly, these data can support HAR, HAU, and HARn tasks. The distinction between these tasks lies in their objectives: HAR focuses on single-label classification, HAU generates detailed captions, and HARn predicts the next action within a spatiotemporal context.  
![](images/31f9a736c2c954eb495f70fc793ff13699f2ea92c8d3fe459db4cb85ea9ce53f.jpg)  
Figure 12: Environment Visualization (The left room is Room 1 and the right one is Room 2). Layout with room-wise visual annotations (Bedroom, Kitchen, Bathroom, and Living Room) showing corresponding example images and sensor placements. The icon indicates the location of the ambient sensor.

## C.2 Scope Limitations

CUHK-X has controlled a dataset and benchmarks for multimodal perception, language grounding, and action reasoning in everyday indoor scenes. The current release covers 30 participants aged 20–23 and two indoor environments. Therefore, we do not claim population-level or broad ecological diver sity in CUHK-X. Our main goal is to provide synchronized modalities (RGB/depth/IR/thermal/skeleton/IMU/mmWave)

with scene-level ground truth that stress-test cross-modal grounding and temporal reasoning. Expanding demographics and environments is planned future work.

## C.3 Verification Procedures

In this release, we do not report inter-rater reliability (IRR), e.g., Cohen’s � for binary checklist items or Krippendorf’s � for ordinal coherence ratings, because the final annotations are consensus labels produced after adjudication, making post-hoc IRR on the gold set uninformative. Although the intended tasks (HAR, HAU, HARn) primarily target scene semantics and temporal dependencies, we view formal IRR as valuable for auditability; in future releases, we plan to report IRR on a double-coded holdout and to release the rubric and adjudication protocol to facilitate external review.

![](images/530ddcd0e34f238439e079ff0f86e36bd42c0e1422133a5eb3083caab3a7b531.jpg)  
Figure 13: Data Statistics of CUHK-X.

## C.4 More Actions and Modalities.

A key future direction for CUHK-X is addressing its scalabil ity across actions and modalities. Firstly, while CUHK-X is comprehensive, it could be expanded to include more actions that involve interactions between multiple participants. In addition, while the current version includes data from seven modalities, future expansions could incorporate other modal ities, such as audio, tactile sensors, heart rate, or EEG. These additional modalities would provide deeper insights into human actions by capturing complementary information, such as emotional states, physiological responses, or fine-grained tactile interactions, enriching the dataset’s multimodal na ture. Moreover, while CUHK-X currently includes data from 30 participants, expanding to a larger, more diverse pool of individuals is critical for improving generalizability.

## C.5 More Discussions

C.5.1 Cross-subject and cross-domainHAR. Real deployments inevitably face domain shifts that go beyond subject identity. In CUHK-X, shifts arise from (i) environment (diferent apart ments and room layouts), (ii) sensor placements and view angles, (iii) lighting and thermal conditions, (iv) background clutter and occlusions, and (v) subject attire and execution styles. Although we have reported cross-subject degradation in §6.1.3, which indicates sensitivity to cross-domain, i.e., dif ferent physical environment factors, removing cross-domain data yields a best result of56.56% on RGB. This finding high lights that domain shifts remain a primary bottleneck even for vision-based HAR. Nevertheless, we believe it is feasi ble to improve model robustness for both cross-subject and cross-domain HAR.

C.5.2 Discussion of zero-shot and fine-tuning. Our HAU and HARn evaluations intentionally use zero-shot LVLMs with task-specific prompts to expose modality gaps and avoid confounds from small-scale fine-tuning. This design choice surfaces two realities: (i) current LVLMs are primarily optimized for RGB, and (ii) reasoning over non-RGB modalities remains challenging without targeted adaptation. At the same time, our HAR results show that task-specific fine-tuning on CUHK-X materially improves recognition compared to of-the-shelf backbones (average 76.52% across modalities), motivating a nuanced view of zero-shot vs fine-tuning. In this version, our goal is that CUHK-X can be a benchmark tool to evaluate the performance of diferent LVLMs.

## C.6 Broader Impact of CUHK-X

We hope that CUHK-X can make a meaningful impact across several fields. First, CUHK-X can serve as a benchmark to support conventional HAR algorithms, including, but not limited to, evaluating multimodal algorithms, cross-subject approaches, and cross-domain methods. Additionally, it provides a benchmark for assessing the capabilities of current LLMs in action understanding and reasoning. Second, CUHK-X ofers synchronous multimodal sensor data, making it easier for researchers and practitioners to explore and work with various sensors and tools. This feature makes it a valuable educational resource, serving as a standard dataset for teaching essential topics such as sensor fusion, data annotation, and multimodal reasoning.

Walking  
Eating  
Drinking Water  
Pouring a Drink  
![](images/fd11e37f6df745a9e2d9be0de6848dc15e01e80dcce94c281cd55d803959c222.jpg)  
Figure 14: Visualization on ambient and wearable sensor data.