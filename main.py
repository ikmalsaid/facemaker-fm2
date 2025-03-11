import os
import cv2
import time
import uuid
import colorsys
import requests
import tempfile
import numpy as np
import onnxruntime
from tqdm.auto import tqdm
from datetime import datetime
from colorpaws import ColorPaws
from scripts.face_analyser import Face
from moviepy.editor import VideoFileClip
from scripts import face_analyser, face_utils, face_store, face_swapper, face_enhancer

class FacemakerFM2:
    """Copyright (C) 2025 Ikmal Said. All rights reserved"""
    
    fm_models = {
        'face_recognizer'   : 'models/arcface_w600k_r50.ism',
        'face_detector'     : 'models/yoloface_8n.ism',
        'face_swapper'      : {
            'fp32'          : 'models/facemaker_fp32.ism',
            'fp16'          : 'models/facemaker_fp16.ism'
        }, 
        'face_landmarker'   : {
            '2dfan4'        : 'models/2dfan4.ism',
            'peppa_wutz'    : 'models/peppa_wutz.ism'
        },
        'face_enhancer'     : {
            'codeformer'    : 'models/codeformer.ism',
            'gfpgan_1.3'    : 'models/gfpgan_1.3.ism',
            'gfpgan_1.4'    : 'models/gfpgan_1.4.ism',
            'gpen-bfr-512'  : 'models/gpen_bfr_512.ism',
            'restoreformer' : 'models/restoreformer.ism'
        }
    }

    def __init__(self, mode='default', skip_enhancer=False, face_swapper_precision='fp32', face_landmarker='peppa_wutz', face_enhancer='gfpgan_1.4',
                 face_detector_score=0.799, reference_face_distance=0.5, save_format='webp', save_to=None, log_on=True, log_to=None):
        """
        Initialize FacemakerClient

        Parameters:
            mode (str): Set startup mode ('default'|'api'|'webui')
            skip_enhancer (bool): If True, skip face_enhancer
            face_swapper_precision (str): Set face_swapper precision ('fp32'|'fp16')
            face_landmarker (str): Set face_landmarker model ('2dfan4'|'peppa_wutz')
            face_enhancer (str): Set face_enhancer model ('gfpgan_1.4'|'gfpgan_1.3'|'restoreformer'|'codeformer')
            face_detector_score (float): Set face_detector_score (0.0 to 1.0)
            reference_face_distance (float): Set reference_face_distance (0.0 to 1.0)
            model_dir (str): Set model_dir (path to models)
            save_format (str): Set save_format ('webp'|'jpg'|'png')
            save_to (str): Set save_to (path to save processed files). If None, uses temp directory
            log_on (bool): If True, log to console
            log_to (str): If not None, log to file
        """
        self.logger = ColorPaws(self.__class__.__name__, log_on, log_to)
 
        self.face_swapper_precision = face_swapper_precision
        self.face_landmarker = face_landmarker
        self.face_enhancer = face_enhancer
        self.face_detector_score = face_detector_score
        self.reference_face_distance = reference_face_distance
        self.save_format = save_format
        self.save_to = save_to if save_to else tempfile.gettempdir()
        self.skip_enhancer = skip_enhancer
        self.startup_mode = mode
    
        self.models = self.__download_models()
        self.nets = self.__load_models(skip_enhancer)

        self.logger.info(f"{self.__class__.__name__} is ready!")
        self.logger.info("-" * 40)

        if self.startup_mode != 'default':
            self.__startup_mode(self.startup_mode)

    def __startup_mode(self, mode):
        """Changes the mode of the client.
        
        Parameters:
            mode: The mode to change to ('default'|'api'|'webui')
        """
        if mode == 'api':
            self.start_api()

        elif mode == 'webui':
            self.start_webui()

        else:
            raise ValueError(f"Invalid startup mode: {mode}")

    def start_api(self, host: str = "0.0.0.0", port: int = 3223, debug: bool = False):
        """
        Start API server with all endpoints.

        Parameters:
        - host (str): Host to run the server on (default: "0.0.0.0")
        - port (int): Port to run the server on (default: 3223)
        - debug (bool): Enable Flask debug mode (default: False)
        """
        from api import FacemakerWebAPI
        FacemakerWebAPI(self, host=host, port=port, debug=debug)
        
    def start_webui(self, host: str = "0.0.0.0", port: int = 3225, browser: bool = True,
                    upload_size: str = "10MB", public: bool = False, limit: int = 10):
        """
        Start WebUI with all features.
        
        Parameters:
        - host (str): Server host (default: "0.0.0.0")
        - port (int): Server port (default: 3225) 
        - browser (bool): Launch browser automatically (default: True)
        - upload_size (str): Maximum file size for uploads (default: "4MB")
        - public (bool): Enable public URL mode (default: False)
        - limit (int): Maximum number of concurrent requests (default: 10)
        """
        from webui import FacemakerWebUI
        FacemakerWebUI(self, host=host, port=port, browser=browser, upload_size=upload_size, public=public, limit=limit)

    def __download_models(self):
        """Downloads models from the given URL."""
        
        def download_file(filename):
            """Helper function to download a file"""
            file_path = os.path.join('models', filename)
            
            # Skip if file exists and is valid
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                self.logger.info(f"Skipping {filename} - file already exists")
                return True
                
            url = f'https://huggingface.co/ikmalsaid/facemaker/resolve/main/models/{filename}?download=true'
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as f, \
                 tqdm(total=total_size, unit='iB', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        size = f.write(chunk)
                        pbar.update(size)
            
            # Verify file was downloaded successfully
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                return True
            else:
                raise Exception(f"Downloaded file {filename} appears to be empty or invalid")

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

        # Flatten the nested dictionary structure
        model_files = []
        for model_name, value in self.fm_models.items():
            if isinstance(value, dict):
                if model_name == 'face_swapper':
                    # Only include the specified precision model
                    model_files.append(value[self.face_swapper_precision])
                else:
                    model_files.extend(value.values())
            else:
                model_files.append(value)

        # Remove 'models/' prefix from paths
        model_files = [os.path.basename(path) for path in model_files]
        
        for filename in model_files:
            try:
                download_file(filename)
            
            except requests.exceptions.RequestException:
                raise Exception(f"Failed to download required model: {filename}")

    def __load_models(self, skip_enhancer=False, load_model=None):
        """Loads ONNX models for face processing.
        
        Parameters:
            skip_enhance: Whether to skip loading the face enhancement model
            load_model: Specific model to load ('face_landmarker' or 'face_enhancer'). If None, loads all models.
        """
        providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']

        self.session_options = onnxruntime.SessionOptions()
        self.session_options.log_severity_level = 3

        if not hasattr(self, 'nets'):
            self.nets = {}

        models_to_process = {load_model: self.fm_models[load_model]} if load_model else self.fm_models

        for model_name, model_info in models_to_process.items():
            if model_name == 'face_enhancer' and skip_enhancer:
                self.logger.info(f"Skipping {model_name} model!")
                continue
                
            if model_name == 'face_landmarker':
                model_path = model_info[self.face_landmarker]
            
            elif model_name == 'face_enhancer':
                model_path = model_info[self.face_enhancer]
            
            elif model_name == 'face_swapper':
                model_path = model_info[self.face_swapper_precision]
            
            else:
                model_path = model_info
                
            self.nets[model_name] = onnxruntime.InferenceSession(
                model_path, 
                sess_options=self.session_options,
                providers=providers
            )
            
            self.logger.info(f"Model loaded: {model_path}")

        # Load model matrix only during initial full load
        if not load_model and 'model_matrix' not in self.nets:
            self.nets['model_matrix'] = np.load('scripts/model_matrix.npy')
            self.nets['is_onnx'] = True

        return self.nets

    def __process_reference_faces(self, source_paths, target_path, source_face_index, target_face_index):
        """Processes and stores reference faces for swapping.
        
        Parameters:
            source_paths: Paths to source face images
            target_path: Path to target image
            source_face_index: Index of face to use from source
            target_face_index: Index of face to replace in target
        """
        source_face = face_analyser.get_average_face(
            face_utils.read_static_images(source_paths), self.nets, self.face_detector_score, position=source_face_index
        )
        reference_frame = face_utils.read_static_image(target_path)
        reference_face = face_analyser.get_one_face(reference_frame, self.nets, self.face_detector_score, position=target_face_index)
        
        face_store.append_reference_face('origin', reference_face)
        
        if source_face and reference_face:
            for processor in ['face_swapper', 'face_enhancer']:
                if processor == 'face_enhancer' and self.skip_enhancer:
                    continue
                if processor in self.nets:
                    get_reference_frame = getattr(globals()[processor], 'get_reference_frame')
                    abstract_frame = get_reference_frame(source_face, reference_face, reference_frame, self.nets)
                    if np.any(abstract_frame):
                        reference_frame = abstract_frame
                        reference_face = face_analyser.get_one_face(reference_frame, self.nets, self.face_detector_score, position=target_face_index)
                        face_store.append_reference_face(processor, reference_face)

    def __process_image(self, source_paths, target_path, enhance_only=False):
        """Performs face swapping and enhancement on a single image.
        
        Parameters:
            source_paths: Paths to source face images
            target_path: Path to target image
            enhance_only: If True, only enhance the target image
        """
        if not enhance_only:
            result = face_swapper.process_image(
                source_paths, target_path, self.reference_face_distance, self.nets, self.face_detector_score)
            
            if not self.skip_enhancer:
                result = face_enhancer.process_image(
                    source_paths, result, self.reference_face_distance, self.nets, self.face_detector_score)
            else:
                result = result
        else:
            result = face_enhancer.process_image(
                source_paths, target_path, self.reference_face_distance, self.nets, self.face_detector_score)
        
        face_swapper.post_process()
        return result

    def get_taskid(self):
        """
        Generate a unique task ID for request tracking.
        Returns a combination of timestamp and UUID to ensure uniqueness.
        Format: YYYYMMDD_HHMMSS_UUID8
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        uuid_part = str(uuid.uuid4())[:8]
        task_id = f"{timestamp}_{uuid_part}"
        return task_id
    
    def __get_savepath(self, target_path, video=False, suffix='_swapped'):
        """Generates output file path based on input arguments.
        
        Parameters:
            target_path: Source file path
            video = If True, set format to 'mp4'
            suffix: Optional suffix for output filename (default '_swapped')
        """
        task_id = self.get_taskid()

        if self.startup_mode == 'api':
            src_name = os.path.splitext(os.path.basename(target_path).split('_')[-1])[0]
        else:
            src_name = os.path.splitext(os.path.basename(target_path))[0]

        if video:
            final_ext = f'.mp4'
        else:
            final_ext = f'.{self.save_format.lower()}'
        
        if self.save_to:
            save_path = os.path.join(self.save_to, f"{task_id}_{src_name}{suffix}{final_ext}")
        else:
            save_path = f"{task_id}_{src_name}{suffix}{final_ext}"

        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        return save_path

    def recognize_from_video(self, target_paths, source_paths, source_face_index=0, target_face_index=0, swap_all_faces=True, detect_direction='left-right', return_output=False):
        """Processes videos by swapping faces in each frame.
        
        Parameters:
            target_paths: Path or list of paths to target videos
            source_paths: Path or list of paths to source face images
            source_face_index: Index of face to use from source
            target_face_index: Index of face to replace in target
            swap_all_faces: Whether to swap all detected faces
            detect_direction: Set the direction of the faces ('left-right'|'right-left'|'top-bottom'|'bottom-top'|'small-large'|'large-small'|'best-worst'|'worst-best')
            return_output: If True, return the processed video path
        """
        # Convert single paths to lists if necessary
        target_paths = [target_paths] if isinstance(target_paths, str) else target_paths
        source_paths = [source_paths] if isinstance(source_paths, str) else source_paths
        
        start_time = time.time()
        timings = {'pre_process': 0, 'face_detection': 0, 'process_reference': 0, 'process_image': 0}
        frames_processed = 0
        save_paths = []  # Initialize list to collect all save paths
        
        for target_path in target_paths:
            self.logger.info(f'Processing video: {target_path}')
            
            t0 = time.time()
            face_swapper.pre_process(source_paths, self.nets, self.face_detector_score)
            timings['pre_process'] += time.time() - t0
            
            video = VideoFileClip(target_path)
            
            def process_frame(frame):
                nonlocal frames_processed, timings
                frames_processed += 1
                try:
                    frame_bgr = frame[..., ::-1].copy()
                    
                    if swap_all_faces:
                        t0 = time.time()
                        all_faces = face_analyser.get_many_faces(frame_bgr, self.nets, self.face_detector_score, detect_direction)
                        timings['face_detection'] += time.time() - t0
                        
                        if not all_faces:
                            self.logger.warning("No faces detected in the target frame!")
                            return frame
                        
                        result_frame = frame_bgr.copy()
                        
                        for target_index, _ in enumerate(all_faces):
                            t0 = time.time()
                            self.__process_reference_faces(source_paths, frame_bgr, source_face_index, target_index)
                            timings['process_reference'] += time.time() - t0
                            
                            t0 = time.time()
                            result_frame = self.__process_image(source_paths, result_frame)
                            timings['process_image'] += time.time() - t0
                        
                    else:
                        t0 = time.time()
                        self.__process_reference_faces(source_paths, frame_bgr, source_face_index, target_face_index)
                        timings['process_reference'] += time.time() - t0
                        
                        t0 = time.time()
                        result_frame = self.__process_image(source_paths, frame_bgr)
                        timings['process_image'] += time.time() - t0

                    return result_frame[..., ::-1]

                except Exception as e:
                    self.logger.error(f"Error processing frame: {str(e)}")
                    return frame
            
            processed_video = video.fl_image(process_frame)
            
            save_path = self.__get_savepath(target_path, video=True)
            processed_video.write_videofile(save_path, 
                                         codec='libx264', 
                                         audio_codec='aac',
                                         fps=video.fps)
            video.close()
            processed_video.close()
            save_paths.append(save_path)  # Add save_path to list
            self.logger.info(f'Saved processed video to: {save_path}')
        
        total_time = time.time() - start_time
        
        # Print timing summary
        self.logger.info("-" * 40)
        self.logger.info("Video Face Swapping Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total frames processed: {frames_processed}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average time per frame: {total_time/frames_processed:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Operation Breakdown:")
        self.logger.info("-" * 40)
        for operation, duration in timings.items():
            percentage = (duration / total_time) * 100
            self.logger.info(f"{operation:20} ({duration:6.2f}s) ({percentage:5.1f}%)")
        self.logger.info("-" * 40)

        if return_output:
            return save_paths

    def recognize_from_image(self, target_paths, source_paths, source_face_index=0, target_face_index=0, swap_all_faces=False, detect_direction='left-right', return_output=False):
        """Processes images by swapping faces.
        
        Parameters:
            target_paths: Path or list of paths to target images
            source_paths: Path or list of paths to source face images
            source_face_index: Index of face to use from source
            target_face_index: Index of face to replace in target
            swap_all_faces: Whether to swap all detected faces
            detect_direction: Set the direction of the faces ('left-right'|'right-left'|'top-bottom'|'bottom-top'|'small-large'|'large-small'|'best-worst'|'worst-best')
            return_output: If True, return the processed image path
        """
        # Convert single paths to lists if necessary
        target_paths = [target_paths] if isinstance(target_paths, str) else target_paths
        source_paths = [source_paths] if isinstance(source_paths, str) else source_paths
        
        start_time = time.time()
        timings = {'pre_process': 0, 'face_detection': 0, 'process_reference': 0, 'process_image': 0, 'image_write': 0}
        images_processed = 0
        save_paths = []  # List to store all output paths

        for target_path in target_paths:
            images_processed += 1
            self.logger.info(f'Processing image: {target_path}')
            
            t0 = time.time()
            face_swapper.pre_process(source_paths, self.nets, self.face_detector_score)
            timings['pre_process'] += time.time() - t0
            
            if swap_all_faces:
                reference_frame = face_utils.read_static_image(target_path)
                
                t0 = time.time()
                all_faces = face_analyser.get_many_faces(reference_frame, self.nets, self.face_detector_score, detect_direction)
                timings['face_detection'] += time.time() - t0
                
                if not all_faces:
                    self.logger.warning("No faces detected in the target image!")
                    return
                
                for target_index, _ in enumerate(all_faces):
                    t0 = time.time()
                    self.__process_reference_faces(source_paths, target_path, source_face_index, target_index)
                    timings['process_reference'] += time.time() - t0
                    
                    t0 = time.time()
                    result_image = self.__process_image(source_paths, reference_frame)
                    timings['process_image'] += time.time() - t0
                    reference_frame = result_image.copy()
                
                save_path = self.__get_savepath(target_path)
                t0 = time.time()
                face_utils.write_image(save_path, reference_frame)
                timings['image_write'] += time.time() - t0
                save_paths.append(save_path)
                self.logger.info(f'Saved final result to: {save_path}')
                
            else:
                t0 = time.time()
                self.__process_reference_faces(source_paths, target_path, source_face_index, target_face_index)
                timings['process_reference'] += time.time() - t0
                
                t0 = time.time()
                result_image = self.__process_image(source_paths, target_path)
                timings['process_image'] += time.time() - t0
                
                save_path = self.__get_savepath(target_path)
                t0 = time.time()
                face_utils.write_image(save_path, result_image)
                timings['image_write'] += time.time() - t0
                save_paths.append(save_path)
                self.logger.info(f'Saved result to: {save_path}')
        
        total_time = time.time() - start_time
        
        # Print timing summary
        self.logger.info("-" * 40)
        self.logger.info("Image Face Swapping Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total images processed: {images_processed}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average time per image: {total_time/images_processed:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Operation Breakdown:")
        self.logger.info("-" * 40)
        for operation, duration in timings.items():
            percentage = (duration / total_time) * 100
            self.logger.info(f"{operation:20} ({duration:6.2f}s) ({percentage:5.1f}%)")
        self.logger.info("-" * 40)

        if return_output:
            return save_paths

    def enhance_from_image(self, target_paths, target_face_index=0, enhance_all_faces=True, detect_direction='left-right', return_output=False):
        """Enhance faces in an image without swapping.
        
        Parameters:
            target_paths: Path or list of paths to target images
            target_face_index: Index of face to enhance (default 0)
            enhance_all_faces: Whether to enhance all detected faces
            detect_direction: Set the direction of the faces ('left-right'|'right-left'|'top-bottom'|'bottom-top'|'small-large'|'large-small'|'best-worst'|'worst-best')
            return_output: If True, returns path to enhanced image
        """
        start_time = time.time()
        timings = {
            'process_reference': 0,
            'process_image': 0, 
            'image_write': 0
        }
        save_paths = []
        images_processed = 0

        if not isinstance(target_paths, (list, tuple)):
            target_paths = [target_paths]

        for path in target_paths:
            images_processed += 1
            reference_frame = face_utils.read_static_image(path)
            self.logger.info(f'Processing image: {path}')
            
            if enhance_all_faces:
                all_faces = face_analyser.get_many_faces(reference_frame, self.nets, self.face_detector_score, detect_direction)
                
                if not all_faces:
                    self.logger.warning("No faces detected in the target image!")
                    continue
                
                for target_index, _ in enumerate(all_faces):
                    t0 = time.time()
                    reference_face = face_analyser.get_one_face(reference_frame, self.nets, self.face_detector_score, position=target_index)
                    face_store.append_reference_face('origin', reference_face)
                    timings['process_reference'] += time.time() - t0

                    if reference_face and 'face_enhancer' in self.nets:
                        t0 = time.time()
                        result_image = self.__process_image(None, reference_frame, enhance_only=True)
                        timings['process_image'] += time.time() - t0
                        reference_frame = result_image.copy()
                
                save_path = self.__get_savepath(path, suffix='_enhanced')
                t0 = time.time()
                face_utils.write_image(save_path, reference_frame)
                timings['image_write'] += time.time() - t0
                save_paths.append(save_path)
                self.logger.info(f'Saved enhanced result to: {save_path}')
            
            else:
                t0 = time.time()
                reference_face = face_analyser.get_one_face(reference_frame, self.nets, self.face_detector_score, position=target_face_index)
                face_store.append_reference_face('origin', reference_face)
                timings['process_reference'] += time.time() - t0

                if reference_face and 'face_enhancer' in self.nets:
                    t0 = time.time()
                    reference_frame = self.__process_image(None, reference_frame, enhance_only=True)
                    timings['process_image'] += time.time() - t0

                    save_path = self.__get_savepath(path, suffix='_enhanced')
                    t0 = time.time()
                    face_utils.write_image(save_path, reference_frame)
                    timings['image_write'] += time.time() - t0
                    save_paths.append(save_path)
                    self.logger.info(f'Saved enhanced result to: {save_path}')

        total_time = time.time() - start_time
        
        # Print timing summary
        self.logger.info("-" * 40)
        self.logger.info("Face Enhancement Summary:")
        self.logger.info("-" * 40) 
        self.logger.info(f"Total images processed: {images_processed}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average time per image: {total_time/images_processed:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Operation Breakdown:")
        self.logger.info("-" * 40)
        for operation, duration in timings.items():
            percentage = (duration / total_time) * 100
            self.logger.info(f"{operation:20} ({duration:6.2f}s) ({percentage:5.1f}%)")
        self.logger.info("-" * 40)

        if return_output:
            return save_paths

    def enhance_from_video(self, target_paths, target_face_index=0, enhance_all_faces=True, detect_direction='left-right', return_output=False):
        """Enhance faces in videos without swapping.
        
        Parameters:
            target_paths: Path or list of paths to target videos
            target_face_index: Index of face to enhance (default 0)
            enhance_all_faces: Whether to enhance all detected faces
            detect_direction: Set the direction of the faces ('left-right'|'right-left'|'top-bottom'|'bottom-top'|'small-large'|'large-small'|'best-worst'|'worst-best')
            return_output: If True, returns path to enhanced video
        """
        # Convert single paths to lists if necessary
        target_paths = [target_paths] if isinstance(target_paths, str) else target_paths
        
        start_time = time.time()
        timings = {'face_detection': 0, 'process_reference': 0, 'process_image': 0}
        frames_processed = 0
        save_paths = []  # Initialize list to collect all save paths
        
        for target_path in target_paths:
            self.logger.info(f'Processing video: {target_path}')
            video = VideoFileClip(target_path)
            
            def process_frame(frame):
                nonlocal frames_processed, timings
                frames_processed += 1
                try:
                    frame_bgr = frame[..., ::-1].copy()
                    
                    if enhance_all_faces:
                        t0 = time.time()
                        all_faces = face_analyser.get_many_faces(frame_bgr, self.nets, self.face_detector_score, detect_direction)
                        timings['face_detection'] += time.time() - t0
                        
                        if not all_faces:
                            return frame
                        
                        result_frame = frame_bgr.copy()
                        
                        for target_index, _ in enumerate(all_faces):
                            t0 = time.time()
                            reference_face = face_analyser.get_one_face(result_frame, self.nets, self.face_detector_score, position=target_index)
                            face_store.append_reference_face('origin', reference_face)
                            timings['process_reference'] += time.time() - t0
                            
                            if reference_face and 'face_enhancer' in self.nets:
                                t0 = time.time()
                                result_frame = self.__process_image(None, result_frame, enhance_only=True)
                                timings['process_image'] += time.time() - t0
                    
                    else:
                        t0 = time.time()
                        reference_face = face_analyser.get_one_face(frame_bgr, self.nets, self.face_detector_score, position=target_face_index)
                        face_store.append_reference_face('origin', reference_face)
                        timings['process_reference'] += time.time() - t0
                        
                        if reference_face and 'face_enhancer' in self.nets:
                            t0 = time.time()
                            result_frame = self.__process_image(None, frame_bgr, enhance_only=True)
                            timings['process_image'] += time.time() - t0
                        else:
                            result_frame = frame_bgr

                    return result_frame[..., ::-1]

                except Exception as e:
                    self.logger.error(f"Error processing frame: {str(e)}")
                    return frame
            
            processed_video = video.fl_image(process_frame)
            
            save_path = self.__get_savepath(target_path, video=True, suffix='_enhanced')
            processed_video.write_videofile(save_path, 
                                         codec='libx264', 
                                         audio_codec='aac',
                                         fps=video.fps)
            video.close()
            processed_video.close()
            save_paths.append(save_path)
            self.logger.info(f'Saved enhanced video to: {save_path}')
        
        total_time = time.time() - start_time
        
        # Print timing summary
        self.logger.info("-" * 40)
        self.logger.info("Video Face Enhancement Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total frames processed: {frames_processed}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average time per frame: {total_time/frames_processed:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Operation Breakdown:")
        self.logger.info("-" * 40)
        for operation, duration in timings.items():
            percentage = (duration / total_time) * 100
            self.logger.info(f"{operation:20} ({duration:6.2f}s) ({percentage:5.1f}%)")
        self.logger.info("-" * 40)

        if return_output:
            return save_paths

    def get_faces_from_image(self, image_path, thumbnail=False, return_marked_image=False, detect_direction='left-right', 
                             unique_faces=True, similarity_threshold=0.4):
        """Find multiple faces in an image and return a list of Face objects.
        
        Parameters:
            image_path: Path to the image file
            thumbnail: If True, also returns face thumbnails as tuple of (image, label)
            return_marked_image: If True, also returns the original image with marked faces
            detect_direction: Set the direction of the faces ('left-right'|'right-left'|'top-bottom'|'bottom-top'|'small-large'|'large-small'|'best-worst'|'worst-best')
            unique_faces: If True, attempts to identify and group similar faces
            similarity_threshold: Threshold for considering faces as the same person (0.0 to 1.0)
        """
        start_time = time.time()
        timings = {
            'image_load': 0,
            'face_detection': 0,
            'thumbnail_creation': 0,
            'face_marking': 0,
            'face_grouping': 0
        }
        
        t0 = time.time()
        self.logger.info(f'Processing: {image_path}')
        src_image = face_utils.read_static_image(image_path)
        marked_image = src_image.copy()
        timings['image_load'] = time.time() - t0
        
        t0 = time.time()
        faces_list = face_analyser.get_many_faces(src_image, self.nets, self.face_detector_score, detect_direction)
        timings['face_detection'] = time.time() - t0
        total_faces = len(faces_list)
        
        # Group similar faces if requested
        unique_face_groups = []
        unique_face_thumbnails = []
        
        if unique_faces and total_faces > 0:
            t0 = time.time()
            # Initialize with first face
            unique_face_groups = [[0]]  # List of lists, each inner list contains indices of similar faces
            
            # Compare each face with the first face of each group
            for i in range(1, total_faces):
                found_match = False
                for group_idx, group in enumerate(unique_face_groups):
                    # Compare with the first face in the group
                    reference_face_idx = group[0]
                    similarity = 1 - face_analyser.calc_face_distance(
                        faces_list[i], 
                        faces_list[reference_face_idx]
                    )
                    
                    if similarity > similarity_threshold:
                        # Add to existing group
                        unique_face_groups[group_idx].append(i)
                        found_match = True
                        break
                        
                if not found_match:
                    # Create new group
                    unique_face_groups.append([i])
        
            unique_face_count = len(unique_face_groups)
            self.logger.info(f"Grouped {total_faces} faces into {unique_face_count} unique individuals")
            timings['face_grouping'] = time.time() - t0
        else:
            unique_face_count = total_faces
        
        # Mark faces on the image
        t0 = time.time()
        for idx, face in enumerate(faces_list):
            # Determine group for this face if using unique faces
            group_id = None
            if unique_faces:
                for group_idx, group in enumerate(unique_face_groups):
                    if idx in group:
                        group_id = group_idx
                        break
            
            # Draw bounding box (use different colors for different groups)
            x1, y1, x2, y2 = map(int, face.bounding_box)
            if unique_faces:
                # Generate a distinct color for each group
                color_hue = (group_id * 137) % 360  # Golden ratio to get well-distributed colors
                color_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(color_hue/360, 0.8, 0.9))
                color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert RGB to BGR
            else:
                color_bgr = (0, 255, 0)  # Default green
            
            cv2.rectangle(marked_image, (x1, y1), (x2, y2), color_bgr, 2)
            
            # Calculate appropriate font scale based on face size
            face_width = x2 - x1
            font_scale = max(0.3, min(0.75, face_width / 200))
            thickness = max(1, min(2, int(face_width / 150)))
            
            # Draw face index and score inside bottom of box
            if unique_faces:
                score_text = f"person_{group_id} ({face.score:.2f})"
            else:
                score_text = f"face_{idx} ({face.score:.2f})"
            
            # Get text size to properly position it
            (text_width, text_height), baseline = cv2.getTextSize(
                score_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
            )
            # Calculate text position (centered horizontally, near bottom of box)
            text_x = x1 + (x2 - x1 - text_width) // 2
            text_y = y2 - baseline  # Place text above bottom edge with baseline offset
            
            overlay = marked_image.copy()
            cv2.rectangle(overlay, 
                         (text_x - 5, text_y - text_height - 5),
                         (text_x + text_width + 5, text_y + 5),
                         (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, marked_image, 0.5, 0, marked_image)
            
            cv2.putText(marked_image, score_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, font_scale, color_bgr, thickness)
            
            # Adjust landmark size based on face size
            landmark_size_68 = max(1, int(face_width / 300))
            landmark_size_5 = max(1, int(face_width / 100))
            
            landmarks_68 = face.landmark['68']
            for x, y in landmarks_68:
                x, y = int(x), int(y)
                cv2.circle(marked_image, (x, y), landmark_size_68, (0, 0, 255), -1)
            
            landmarks_5 = face.landmark['5']
            for x, y in landmarks_5:
                x, y = int(x), int(y)
                cv2.circle(marked_image, (x, y), landmark_size_5, (255, 0, 0), -1)
        
        marked_image = cv2.cvtColor(marked_image, cv2.COLOR_BGR2RGB)
        timings['face_marking'] = time.time() - t0
        
        if thumbnail:
            thumbnails = []
            t0 = time.time()
            
            if unique_faces:
                # Create one thumbnail per unique face group (using the best face from each group)
                for group_idx, group in enumerate(unique_face_groups):
                    # Find the face with highest score in this group
                    best_face_idx = group[0]
                    best_score = faces_list[best_face_idx].score
                    
                    for face_idx in group:
                        if faces_list[face_idx].score > best_score:
                            best_face_idx = face_idx
                            best_score = faces_list[face_idx].score
                    
                    # Extract thumbnail for best face
                    face = faces_list[best_face_idx]
                    x1, y1, x2, y2 = map(int, face.bounding_box)
                    face_img = src_image[y1:y2, x1:x2]
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    # Create label with person ID
                    label = f"person_{group_idx}"
                    thumbnails.append((face_img_rgb, label))
                    unique_face_thumbnails.append((face_img_rgb, label))
            else:
                # Create thumbnail for each detected face
                for idx, face in enumerate(faces_list):
                    x1, y1, x2, y2 = map(int, face.bounding_box)
                    face_img = src_image[y1:y2, x1:x2]
                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                    # Create label with face index
                    label = f"face_{idx}"
                    thumbnails.append((face_img_rgb, label))
                
            timings['thumbnail_creation'] = time.time() - t0
        
        total_time = time.time() - start_time
        
        # Print timing summary
        self.logger.info("-" * 40)
        self.logger.info("Face Detection Summary:")
        self.logger.info("-" * 40)
        if unique_faces:
            self.logger.info(f"Total faces found: {total_faces}")
            self.logger.info(f"Unique individuals identified: {unique_face_count}")
        else:
            self.logger.info(f"Total faces found: {total_faces}")
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Operation Breakdown:")
        self.logger.info("-" * 40)
        for operation, duration in timings.items():
            percentage = (duration / total_time) * 100
            self.logger.info(f"{operation:20} ({duration:6.2f}s) ({percentage:5.1f}%)")
        self.logger.info("-" * 40)
        
        if thumbnail:
            if unique_faces:
                thumbnails_to_return = unique_face_thumbnails
            else:
                thumbnails_to_return = thumbnails
            
            if return_marked_image:
                return faces_list, unique_face_count if unique_faces else total_faces, thumbnails_to_return, marked_image
            else:
                return faces_list, unique_face_count if unique_faces else total_faces, thumbnails_to_return, None

        if return_marked_image:
            return faces_list, unique_face_count if unique_faces else total_faces, None, marked_image
        else:
            return faces_list, unique_face_count if unique_faces else total_faces, None, None

    def get_faces_from_video(self, video_path, thumbnail=False, return_marked_video=False, detect_direction='left-right',
                             unique_faces=True, sample_rate=10, similarity_threshold=0.8):
        """Find multiple faces in each frame of a video and return face information.
        
        Parameters:
            video_path: Path to the video file
            thumbnail: If True, returns face thumbnails from each frame as tuple of (image, label, frame_number)
            return_marked_video: If True, returns the processed video with marked faces
            detect_direction: Set the direction of the faces ('left-right'|'right-left'|'top-bottom'|'bottom-top'|'small-large'|'large-small'|'best-worst'|'worst-best')
            unique_faces: If True, attempts to track and count only unique faces across frames
            sample_rate: Process only every Nth frame for face detection when unique_faces is True
            similarity_threshold: Threshold for considering faces as the same person (0.0 to 1.0)
        """
        start_time = time.time()
        timings = {
            'frame_load': 0,
            'face_detection': 0,
            'face_marking': 0,
            'thumbnail_creation': 0,
            'face_tracking': 0
        }
        
        frames_processed = 0
        total_faces = 0
        all_faces = []  # List to store faces from all frames
        all_thumbnails = []  # List to store thumbnails if requested
        
        # For tracking unique faces
        unique_face_embeddings = []
        unique_face_thumbnails = []
        unique_face_count = 0
        
        self.logger.info(f'Processing video: {video_path}')
        video = VideoFileClip(video_path)
        
        def process_frame(frame):
            nonlocal frames_processed, total_faces, timings, unique_face_count
            frames_processed += 1
            
            # Skip frames based on sample rate when in unique faces mode
            if unique_faces and frames_processed % sample_rate != 0 and not return_marked_video:
                return frame
            
            try:
                # Convert frame to BGR for OpenCV processing
                t0 = time.time()
                frame_bgr = frame[..., ::-1].copy()
                marked_frame = frame_bgr.copy()
                timings['frame_load'] += time.time() - t0
                
                # Detect faces in the frame
                t0 = time.time()
                faces_list = face_analyser.get_many_faces(frame_bgr, self.nets, self.face_detector_score, detect_direction)
                timings['face_detection'] += time.time() - t0
                
                # Initialize face_group_ids for this frame
                face_group_ids = [-1] * len(faces_list)  # Default to -1 (no group)
                
                # Process detected faces
                if faces_list:
                    # Track unique faces if enabled
                    if unique_faces:
                        t0 = time.time()
                        for idx, face in enumerate(faces_list):
                            is_new_face = True
                            
                            # Compare with existing unique faces
                            for i, unique_embedding in enumerate(unique_face_embeddings):
                                similarity = 1 - face_analyser.calc_face_distance(face, Face(
                                    bounding_box=None,
                                    landmark=None,
                                    score=None,
                                    embedding=None,
                                    normed_embedding=unique_embedding
                                ))
                                
                                if similarity > similarity_threshold:
                                    is_new_face = False
                                    face_group_ids[idx] = i  # Assign to existing group
                                    break
                                
                            if is_new_face:
                                # Create new group
                                unique_face_count += 1
                                unique_face_embeddings.append(face.normed_embedding)
                                face_group_ids[idx] = len(unique_face_embeddings) - 1
                                
                                # Store thumbnail of unique face if requested
                                if thumbnail:
                                    x1, y1, x2, y2 = map(int, face.bounding_box)
                                    face_img = frame_bgr[y1:y2, x1:x2]
                                    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                                    unique_face_thumbnails.append((face_img_rgb, f"person_{face_group_ids[idx]}"))
                        
                        timings['face_tracking'] += time.time() - t0
                    
                    # For regular processing (counting all faces)
                    total_faces += len(faces_list)
                    all_faces.append((frames_processed, faces_list))
                
                    # Mark faces on the frame if return_marked_video is True
                    t0 = time.time()
                    for idx, face in enumerate(faces_list):
                        # Draw bounding box with appropriate color
                        x1, y1, x2, y2 = map(int, face.bounding_box)
                        
                        if unique_faces:
                            # Get the group ID for this face
                            group_id = face_group_ids[idx]
                            
                            # Generate a distinct color for each group
                            color_hue = (group_id * 137) % 360  # Golden ratio to get well-distributed colors
                            color_rgb = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(color_hue/360, 0.8, 0.9))
                            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])  # Convert RGB to BGR
                            
                            # Create label with person ID
                            score_text = f"person_{group_id} ({face.score:.2f})"
                        else:
                            color_bgr = (0, 255, 0)  # Default green
                            score_text = f"face_{idx} ({face.score:.2f})"
                        
                        cv2.rectangle(marked_frame, (x1, y1), (x2, y2), color_bgr, 2)
                        
                        # Calculate appropriate font scale based on face size
                        face_width = x2 - x1
                        font_scale = max(0.3, min(0.75, face_width / 200))
                        thickness = max(1, min(2, int(face_width / 150)))
                        
                        # Get text size to properly position it
                        (text_width, text_height), baseline = cv2.getTextSize(
                            score_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
                        )
                        text_x = x1 + (x2 - x1 - text_width) // 2
                        text_y = y2 - baseline
                        
                        # Add semi-transparent background for text
                        overlay = marked_frame.copy()
                        cv2.rectangle(overlay,
                                    (text_x - 5, text_y - text_height - 5),
                                    (text_x + text_width + 5, text_y + 5),
                                    (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.5, marked_frame, 0.5, 0, marked_frame)
                        
                        # Add text
                        cv2.putText(marked_frame, score_text, (text_x, text_y),
                                  cv2.FONT_HERSHEY_DUPLEX, font_scale, color_bgr, thickness)
                        
                        # Adjust landmark size based on face size
                        landmark_size_68 = max(1, int(face_width / 300))
                        landmark_size_5 = max(1, int(face_width / 100))
                        
                        # Draw landmarks
                        landmarks_68 = face.landmark['68']
                        for x, y in landmarks_68:
                            x, y = int(x), int(y)
                            cv2.circle(marked_frame, (x, y), landmark_size_68, (0, 0, 255), -1)
                        
                        landmarks_5 = face.landmark['5']
                        for x, y in landmarks_5:
                            x, y = int(x), int(y)
                            cv2.circle(marked_frame, (x, y), landmark_size_5, (255, 0, 0), -1)
                        
                        # Create thumbnails if requested and not in unique faces mode
                        if thumbnail and not unique_faces:
                            t0 = time.time()
                            face_img = frame_bgr[y1:y2, x1:x2]
                            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            # Calculate timestamp
                            seconds = frames_processed / video.fps
                            hours = int(seconds // 3600)
                            minutes = int((seconds % 3600) // 60)
                            seconds = int(seconds % 60)
                            timestamp = f"{hours:02d}h{minutes:02d}m{seconds:02d}s"
                            # Create label with face index, frame number and timestamp
                            label = f"face_{idx}_frame_{frames_processed}_{timestamp}"
                            all_thumbnails.append((face_img_rgb, label))
                            timings['thumbnail_creation'] += time.time() - t0
                    
                    timings['face_marking'] += time.time() - t0
                
                # Return marked frame if return_marked_video is True, otherwise return original
                if return_marked_video:
                    return marked_frame[..., ::-1]
                
                return frame
            
            except Exception as e:
                self.logger.error(f"Error processing frame {frames_processed}: {str(e)}")
                return frame
        
        if return_marked_video:
            processed_video = video.fl_image(process_frame)
            marked_video = self.__get_savepath(video_path, video=True, suffix='_marked')
            processed_video.write_videofile(marked_video,
                                         codec='libx264',
                                         audio_codec='aac',
                                         fps=video.fps)
            video.close()
            processed_video.close()
        else:
            # Process frames without saving video
            for frame in video.iter_frames():
                process_frame(frame)
            video.close()
        
        total_time = time.time() - start_time
        
        # Print timing summary
        self.logger.info("-" * 40)
        self.logger.info("Video Face Detection Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Total frames processed: {frames_processed}")
        
        if unique_faces:
            self.logger.info(f"Total unique faces detected: {unique_face_count}")
            self.logger.info(f"Total faces detected (all frames): {total_faces}")
        else:
            self.logger.info(f"Total faces detected: {total_faces}")
            self.logger.info(f"Average faces per frame: {total_faces/frames_processed:.2f}")
        
        self.logger.info(f"Total processing time: {total_time:.2f} seconds")
        self.logger.info(f"Average time per frame: {total_time/frames_processed:.2f} seconds")
        self.logger.info("-" * 40)
        self.logger.info("Operation Breakdown:")
        self.logger.info("-" * 40)
        for operation, duration in timings.items():
            percentage = (duration / total_time) * 100
            self.logger.info(f"{operation:20} ({duration:6.2f}s) ({percentage:5.1f}%)")
        self.logger.info("-" * 40)
        
        if thumbnail:
            if unique_faces:
                thumbnails_to_return = unique_face_thumbnails
            else:
                thumbnails_to_return = all_thumbnails
            
            if return_marked_video:
                return all_faces, unique_face_count if unique_faces else total_faces, thumbnails_to_return, marked_video
            else:
                return all_faces, unique_face_count if unique_faces else total_faces, thumbnails_to_return, None
        
        if return_marked_video:
            return all_faces, unique_face_count if unique_faces else total_faces, None, marked_video
        else:
            return all_faces, unique_face_count if unique_faces else total_faces, None, None

    def change_settings(self, face_swapper_precision=None, face_landmarker=None, face_enhancer=None, face_detector_score=None,
                        reference_face_distance=None, save_format=None, save_to=None, skip_enhancer=False):
        """Update face processing settings.
        
        Parameters:
            face_landmarker_name: Set face_landmarker model ('2dfan4'|'peppa_wutz')
            face_enhancer_name: Set face_enhancer model ('gfpgan_1.4'|'gfpgan_1.3'|'restoreformer'|'codeformer')
            face_detector_score: Set face_detector_score (0.0 to 1.0)
            reference_face_distance: Set reference_face_distance (0.0 to 1.0)
            save_format: Set save_format ('none'|'webp'|'jpg'|'png')
            save_to: Set save_to (path to save the output)
            skip_enhancer: Set skip_enhancer
        """
        if face_swapper_precision is not None and face_swapper_precision != self.face_swapper_precision:
            if face_swapper_precision not in self.fm_models['face_swapper']:
                self.logger.warning("face_swapper_precision must be 'fp32' or 'fp16'! Using 'fp32' as default.")
                face_swapper_precision = 'fp32'
            
            self.face_swapper_precision = face_swapper_precision
            self.__load_models(load_model='face_swapper')

        if face_landmarker is not None and face_landmarker != self.face_landmarker:
            if face_landmarker not in self.fm_models['face_landmarker']:
                self.logger.warning(f"Invalid face_landmarker model name! Using peppa_wutz as default.")
                face_landmarker = 'peppa_wutz'
            
            self.face_landmarker = face_landmarker
            self.__load_models(load_model='face_landmarker')

        if face_enhancer is not None and face_enhancer != self.face_enhancer:
            if face_enhancer not in self.fm_models['face_enhancer']:
                self.logger.warning(f"Invalid face_enhancer model name! Using gfpgan_1.4 as default.")
                face_enhancer = 'gfpgan_1.4'
            
            self.face_enhancer = face_enhancer
            self.__load_models(load_model='face_enhancer')

        if face_detector_score is not None and face_detector_score != self.face_detector_score:
            if not 0 <= face_detector_score <= 1:
                self.logger.warning("face_detector_score must be between 0 and 1! Using 0.5 as default.")
                face_detector_score = 0.5
            
            self.face_detector_score = face_detector_score
            
        if reference_face_distance is not None and reference_face_distance != self.reference_face_distance:
            if not 0 <= reference_face_distance <= 1:
                self.logger.warning("reference_face_distance must be between 0 and 1! Using 0.5 as default.")
                reference_face_distance = 0.5
            
            self.reference_face_distance = reference_face_distance

        if save_format is not None and save_format != self.save_format:
            if save_format not in ['None', 'webp', 'jpg', 'png']:
                self.logger.warning("save_format must be 'None', 'webp', 'jpg', or 'png'! Using 'none' as default.")
                save_format = None
            
            self.save_format = save_format

        if save_to is not None and save_to != self.save_to:
            if save_to == 'None':
                self.save_to = None
            else:
                self.save_to = save_to

        if skip_enhancer or not skip_enhancer and skip_enhancer != self.skip_enhancer:
            self.skip_enhancer = skip_enhancer
            self.__load_models(skip_enhancer=skip_enhancer)

        face_swapper.post_process()        
        self.logger.info("-" * 40)
        self.logger.info("Applied Settings Summary:")
        self.logger.info("-" * 40)
        self.logger.info(f"Save Format: {self.save_format}")
        self.logger.info(f"Save Location: {self.save_to}")
        self.logger.info(f"Face Enhancer: {self.face_enhancer}")
        self.logger.info(f"Skip Enhancer: {self.skip_enhancer}")
        self.logger.info(f"Face Landmarker: {self.face_landmarker}")
        self.logger.info(f"Face Detector Score: {self.face_detector_score}")
        self.logger.info(f"Face Swapper Precision: {self.face_swapper_precision}")
        self.logger.info(f"Reference Face Distance: {self.reference_face_distance}")
        self.logger.info("-" * 40)