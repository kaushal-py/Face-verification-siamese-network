# Cross Verification of Images for Eye care

### Introduction
Sankara Eye Foundation USA is a not-for-profit missionary institution for ophthalmic care. They are Donors who fund cataract surgeries for underprivileged. For this purpose they fund eye hospitals on basis of free cataract surgeries performed. Funds are released, based on the number of operations performed. 

The NGO needs to ensure that the funds are released only for genuine surgeries. As a proof that the surgery is actually performed, the NGO currently requests eye hospitals to send in pre-operation (pre op) and post operation (post op) photographs of the patient’s face for each surgery.

The NGO then verifies the pre op and post op images to ensure that the surgery has actually been performed before releasing funds. They also need to verify that photographs of same patients are not being sent again and again to claim funds fraudulently.

They need a system which will help in comparing these photographs to highlight cases with high probability of fraud. These can then be further investigated manually.

### Problem Statement of the NGO

Pre op and post op photographs will be provided. The NGO needs a system which will help compare the pre and post-operative images and identify that this is the same patient with a high level of confidence. They also want the system to be able to compare the pre op photographs with existing database of pre op images and post op photograph to database of post op images to ensure that the pictures have not been previously submitted.

The post-operative pictures captured, have a part of the patients face covered with a patch. The existing facial recognition algorithms available require all the parts of the face to be visible. Hence when a part of the face is covered with the patch, these algorithms are not very effective.

The problem is to find a solution which will be able to detect and match the patients from pre op and post op images to confirm it is the same patient (even when the part of face is covered with a patch )
The algorithm should also be able to detect or identify if the images are photo shopped.

![Sample Image](http://deepblue.co.in/wp-content/uploads/2017/08/Cross-Verification-example.jpg)

### What the NGO Wants?

The system should be able to recognise and match with a high degree of accuracy the pre and post-operative photographs to check that they are the same person even when the part of the eye in the post op image has a patch.

The case should be highlighted for manual verification if
* The pre op image matches a pre op image in the existing data base
* A post op images matches the post op image existing in database
* The pre and post-operative pictures submitted for the case do not match
* Image is detected as possibly photo shopped.

### The Challenge

The system must be able to match with high level of accuracy that the patient in pre op and post op image is the same person (even with the eye patch). The system must also be able to detect with high level of confidence if the pictures of same patient are submitted twice. The system should also be able to identify images which are photoshoped. The images which look suspicious should be highlighted for further manual investigation

### Setup Instructions

1. Clone the repository
`git clone https://gitlab.com/echodarkstar/EYC3PDBS3.git`

2. Go to the root folder of the repository
`cd EYC3PDBS3`

3. Create conda environment from the yml file
`conda env create -f environment.yml`

> **Note**
> Conda must be installed in your machine to perform this action

4. Activate the Environment
`source activate python35`

### Developed by EYC3

#### Mentor
1. Qushai Dalal

#### Members
1. Nishant Shankar
2. Adheesh Juvekar
3. Asutosh Padhi
4. Kaushal Bhogale
