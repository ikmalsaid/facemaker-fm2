def FacemakerWebAPI(client, host: str = "0.0.0.0", port: int = 3223, debug: bool = False):
    """
    Start Client API server with all endpoints.
    
    Parameters:
    - client (Client): Facemaker Client instance
    - host (str): Host to run the server on
    - port (int): Port to run the server on
    - debug (bool): Enable Flask debug mode
    """
    try:
        from flask import Flask, request, jsonify, render_template, send_file
        import tempfile, os, cv2, mimetypes
        
        app = Flask(__name__)
        app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
        
        def __send_file(file_path):
            """Send a file as binary response with proper mime type.
            
            Parameters:
                file_path: Path to the file to send
            Returns:
                Flask response object with binary file data
            """
            mime_type, _ = mimetypes.guess_type(file_path)
            return send_file(
                file_path,
                mimetype=mime_type,
                as_attachment=True,
                download_name=os.path.basename(file_path)
            )

        @app.route('/')
        def api_docs():
            return render_template('index.html')

        @app.route('/api/download/<path:filename>')
        def download_file(filename):
            """Download endpoint for retrieving processed files"""
            try:
                file_path = os.path.join(tempfile.gettempdir(), filename)
                return __send_file(file_path)
            
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })

        @app.route('/api/swap/image', methods=['POST'])
        def swap_image():
            try:
                task_id = client.get_taskid()
                # Get files from request
                target_files = request.files.getlist('target')
                source_files = request.files.getlist('source')
                
                # Save uploaded files temporarily with task ID prefix
                target_paths = []
                source_paths = []
                
                for file in target_files:
                    temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{file.filename}")
                    file.save(temp_path)
                    target_paths.append(temp_path)
                    
                for file in source_files:
                    temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{file.filename}")
                    file.save(temp_path)
                    source_paths.append(temp_path)
                
                # Get parameters
                source_face_index = int(request.form.get('source_face_index', 0))
                target_face_index = int(request.form.get('target_face_index', 0))
                swap_all = request.form.get('swap_all', 'true').lower() == 'true'
                detect_direction = request.form.get('detect_direction', 'left-right')
                
                # Process images
                result_paths = client.recognize_from_image(
                    target_paths, source_paths,
                    source_face_index, target_face_index,
                    swap_all, detect_direction,
                    return_output=True
                )
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'filename': os.path.basename(path)
                    } for path in result_paths]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        @app.route('/api/swap/video', methods=['POST'])
        def swap_video():
            try:
                task_id = client.get_taskid()
                # Get files from request
                target_files = request.files.getlist('target')
                source_files = request.files.getlist('source')
                
                # Save uploaded files temporarily
                target_paths = []
                source_paths = []
                
                for file in target_files:
                    temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{file.filename}")
                    file.save(temp_path)
                    target_paths.append(temp_path)
                    
                for file in source_files:
                    temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{file.filename}")
                    file.save(temp_path)
                    source_paths.append(temp_path)
                
                # Get parameters
                source_face_index = int(request.form.get('source_face_index', 0))
                target_face_index = int(request.form.get('target_face_index', 0))
                swap_all = request.form.get('swap_all', 'true').lower() == 'true'
                detect_direction = request.form.get('detect_direction', 'left-right')
                
                # Process videos
                result_paths = client.recognize_from_video(
                    target_paths, source_paths,
                    source_face_index, target_face_index,
                    swap_all, detect_direction,
                    return_output=True
                )
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'filename': os.path.basename(path)
                    } for path in result_paths]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        @app.route('/api/detect/image', methods=['POST'])
        def detect_faces():
            try:
                task_id = client.get_taskid()
                # Get image file
                image_file = request.files['image']
                temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{image_file.filename}")
                image_file.save(temp_path)
                
                # Get parameters
                return_thumbnails = request.form.get('return_thumbnails', 'false').lower() == 'true'
                return_marked = request.form.get('return_marked', 'false').lower() == 'true'
                detect_direction = request.form.get('detect_direction', 'left-right')
                
                # Detect faces
                _, total, thumbnails, marked = client.get_faces_from_image(
                    temp_path, return_thumbnails,
                    return_marked, detect_direction
                )
                
                # Save thumbnails and marked image if present
                result = {
                    'total_faces': total
                }
                
                if thumbnails:
                    thumbnail_paths = []
                    for idx, (thumb, label) in enumerate(thumbnails):
                        thumb_path = os.path.join(tempfile.gettempdir(), f"{task_id}_thumb_{idx}_{image_file.filename}")
                        cv2.imwrite(thumb_path, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
                        thumbnail_paths.append({
                            'filename': os.path.basename(thumb_path),
                            'label': label
                        })
                    result['thumbnails'] = thumbnail_paths
                    
                if marked is not None:
                    result['marked_image'] = {
                        'filename': os.path.basename(marked)
                    }
                
                return jsonify({
                    'success': True,
                    'results': result
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
            
        @app.route('/api/detect/video', methods=['POST'])
        def detect_faces_video():
            try:
                task_id = client.get_taskid()
                # Get video file
                video_file = request.files['video']
                temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{video_file.filename}")
                video_file.save(temp_path)
                
                # Get parameters
                return_thumbnails = request.form.get('return_thumbnails', 'false').lower() == 'true'
                return_marked = request.form.get('return_marked', 'false').lower() == 'true'
                detect_direction = request.form.get('detect_direction', 'left-right')
                unique_faces = request.form.get('unique_faces', 'true').lower() == 'true'
                
                # Detect faces
                _, total, thumbnails, marked = client.get_faces_from_video(
                    temp_path, return_thumbnails,
                    return_marked, detect_direction,
                    unique_faces
                )
                
                # Save thumbnails and marked video if present
                result = {
                    'total_faces': total
                }
                
                if thumbnails:
                    thumbnail_paths = []
                    for idx, (thumb, label) in enumerate(thumbnails):
                        thumb_path = os.path.join(tempfile.gettempdir(), f"{task_id}_thumb_{idx}_{video_file.filename}.jpg")
                        cv2.imwrite(thumb_path, cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR))
                        thumbnail_paths.append({
                            'filename': os.path.basename(thumb_path),
                            'label': label
                        })
                    result['thumbnails'] = thumbnail_paths
                    
                if marked is not None:
                    result['marked_video'] = {
                        'filename': os.path.basename(marked)
                    }
                
                return jsonify({
                    'success': True,
                    'results': result
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
            
        @app.route('/api/enhance/image', methods=['POST'])
        def enhance_image():
            try:
                task_id = client.get_taskid()
                # Get files from request
                target_files = request.files.getlist('target')
                
                # Save uploaded files temporarily
                target_paths = []
                for file in target_files:
                    temp_path = os.path.join(tempfile.gettempdir(), f"{task_id}_{file.filename}")
                    file.save(temp_path)
                    target_paths.append(temp_path)
                
                # Get parameters
                target_face_index = int(request.form.get('target_face_index', 0))
                enhance_all = request.form.get('enhance_all', 'true').lower() == 'true'
                detect_direction = request.form.get('detect_direction', 'left-right')
                
                # Process images
                result_paths = client.enhance_from_image(
                    target_paths, target_face_index,
                    enhance_all, detect_direction,
                    return_output=True
                )
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'filename': os.path.basename(path)
                    } for path in result_paths]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        @app.route('/api/enhance/video', methods=['POST'])
        def enhance_video():
            try:
                task_id = client.get_taskid()
                # Get files from request
                target_files = request.files.getlist('target')
                
                # Save uploaded files temporarily
                target_paths = []
                for file in target_files:
                    temp_path = os.path.join(tempfile.gettempdir, f"{task_id}_{file.filename}")
                    file.save(temp_path)
                    target_paths.append(temp_path)
                
                # Get parameters
                target_face_index = int(request.form.get('target_face_index', 0))
                enhance_all = request.form.get('enhance_all', 'true').lower() == 'true'
                detect_direction = request.form.get('detect_direction', 'left-right')
                
                # Process videos
                result_paths = client.enhance_from_video(
                    target_paths, target_face_index,
                    enhance_all, detect_direction,
                    return_output=True
                )
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'filename': os.path.basename(path)
                    } for path in result_paths]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
                
        @app.route('/api/settings', methods=['POST'])
        def update_settings():
            try:
                # Get settings from request with fallback values
                settings = {
                    'face_swapper_precision': request.form.get('face_swapper_precision', 'fp32'),
                    'face_landmarker': request.form.get('face_landmarker', 'peppa_wutz'),
                    'face_enhancer': request.form.get('face_enhancer', 'gfpgan_1.4'),
                    'face_detector_score': float(request.form.get('face_detector_score', '0.799')),
                    'reference_face_distance': float(request.form.get('reference_face_distance', '0.5')),
                    'skip_enhancer': request.form.get('skip_enhancer', 'false') == 'true'
                }
                
                # Update settings
                client.change_settings(**settings)
                
                return jsonify({
                    'success': True,
                    'results': [{
                        'message': 'Settings updated successfully!'
                    }]
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        client.logger.info(f"Starting API server on {host}:{port}")
        app.run(host=host, port=port, debug=debug)
    
    except Exception as e:
        client.logger.error(f"{str(e)}")
        raise