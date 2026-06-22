# AI_Log.md

Log of the AI tools I used while working on this project.

| Date and Time | Prompt | Tool | Response Synopsis | Change |
| --- | --- | --- | --- | --- |
| 06/18/2026  9:40 PM | i'm taking CS 898BA and have to submit a project pitch. can you explain what the pitch needs and what i have to submit? | Claude | Explained the four required pitch sections, the git deliverables, and the video and YouTube submission rules. | Understood the milestone and the submission checklist. |
| 06/18/2026  10:05 PM | help me pick a computer vision project with a good public dataset and real image processing, i'm thinking medical imaging | Claude | Recommended diabetic retinopathy grading on APTOS 2019 since fundus images need real preprocessing. | Picked the project topic. |
| 06/18/2026  10:30 PM | is this diabetic retinopathy idea good enough or too common, and how do i make it strong? | Claude | Said it fits the requirements well but is common; suggested a preprocessing ablation and handling class imbalance to stand out. | Decided to build around a baseline-vs-full-pipeline ablation. |
| 06/19/2026  12:15 PM | give me a short literature review for diabetic retinopathy deep learning, the main papers to cite | Claude | Summarized Gulshan 2016, Pratt 2016, Ben Graham 2015, and EfficientNet (Tan and Le). | Wrote the literature review slide. |
| 06/19/2026  12:50 PM | what model architectures can i use and what are the tradeoffs? need at least two alternatives plus my pick | Claude | Compared a scratch CNN, EfficientNet-B3 transfer learning, and a Vision Transformer. | Defined the three alternative designs. |
| 06/19/2026  1:20 PM | why is EfficientNet-B3 transfer learning better than a scratch CNN or a ViT here? | Claude | Justified on limited data, accuracy per compute, and dataset scale. | Wrote the choice justification. |
| 06/19/2026  7:30 PM | design the image processing pipeline for fundus images, it has to be beyond augmentation, explain each step | Claude | Proposed circular crop, green channel, Ben Graham normalization, CLAHE, top-hat morphology, and denoise. | Built the image processing strategy. |
| 06/19/2026  8:10 PM | what metric should i use for APTOS and how do i deal with the class imbalance? | Claude | Recommended Quadratic Weighted Kappa plus sensitivity, specificity, and a confusion matrix; flagged the imbalance. | Defined the evaluation plan. |
| 06/20/2026  11:00 AM | write python to run the preprocessing steps (green channel, CLAHE, Ben Graham, top-hat) on a fundus image so i can show before and after | Claude | Generated OpenCV and PIL code producing the real processed outputs. | Created the preprocessing demonstration figures. |
| 06/20/2026  2:30 PM | help me build the powerpoint for my pitch, cover all the sections and make it look good | Claude | Helped structure and build the deck with the required sections, figures, and speaker notes. | Built the pitch slide deck. |
| 06/20/2026  4:15 PM | generate some retinal fundus images i can use as illustrations in my slides | Gemini | Generated synthetic fundus image illustrations. | Added the fundus illustrations to the deck. |
| 06/21/2026  10:30 AM | check my pitch against the professor's requirements and tell me what's real data vs illustrative | Claude | Confirmed all required sections present and clarified which figures are real vs illustrative. | Verified rubric alignment before submitting. |
