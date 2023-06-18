# Automated Facial Expression Detection in Sign Language Videos

## Project Definition

The goal of this project is to develop an automated system for detecting facial expressions in sign language videos using a spatio-temporal graph convolutional network (STGCN). The primary application of this system is to aid communication between deaf and hearing individuals.

### Project Steps

1. **Data Collection**: Collect a dataset of sign language videos.
2. **Data Preprocessing**: Use MMPose, an open-source toolkit for 2D human pose estimation, to extract relevant facial landmarks from the sign language videos.
3. **Feature Selection**: Select the key facial landmarks most relevant for detecting facial expressions in sign language.
4. **Graph Construction**: Create a Delaunay graph using the selected facial landmarks.
5. **Graph Representation**: Convert the Delaunay graph into an adjacency matrix to be used as input for training the STGCN model.
6. **Model Training**: Train the STGCN model using the prepared dataset and graph representation.
7. **Evaluation**: Assess the performance of the trained model by measuring accuracy in detecting facial expressions in sign language glosses.
8. **Generalization**: Ensure the trained model can generalize well to different signers and facial features.

## Project Structure

The project directory consists of the following folders:

- **binary**: Contains files related to the training process, including data loaders, filters, and other utilities for preparing the initial data for neagitivity analysis
- **mouth2**: Contains files related to the mouth graphs, including the graph definition, preprocessed data, and trained models specific to mouth features for direct turkish word detection.
- **updated and working**: Contains files related to the face graph, including the graph definition, preprocessed data, and trained models specific to facial features.

Please refer to the respective folders for more details and specific instructions on utilizing the contents.

## Usage

To utilize this project, follow the steps below:

1. Collect a dataset of sign language videos.
2. Preprocess the videos using MMPose to extract relevant facial landmarks.
3. Select the key facial landmarks for detecting facial expressions.
4. Create a Delaunay graph using the selected facial landmarks.
5. Convert the Delaunay graph into an adjacency matrix.
6. Train the STGCN model using the prepared dataset and graph representation.
7. Evaluate the trained model's performance in detecting facial expressions in sign language glosses.
8. Ensure the trained model can generalize well to different signers and facial features.

Please refer to the project documentation and source code for detailed instructions on running the various steps and utilizing the developed system.


## Contact

For any questions, feedback, or inquiries related to the project, please contact:

- Email: [your-email@example.com](mailto:refikakalyoncu@gmail.com)
- GitHub: [YourGitHubUsername](https://github.com/refref18)

Feel free to reach out with any inquiries or contributions to enhance the project.

## References

[1] BosphorusSign22k dataset: [Link to dataset](https://ogulcanozdemir.github.io/bosphorussign22k/)
