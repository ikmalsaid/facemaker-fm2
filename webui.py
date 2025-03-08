def FacemakerWebUI(client, host: str = None, port: int = None, browser: bool = True,
                   upload_size: str = "4MB", public: bool = False, limit: int = 10):
    """ 
    Start Facemaker WebUI with all features.
    
    Parameters:
    - client (Client): Facemaker Client instance
    - host (str): Server host
    - port (int): Server port
    - browser (bool): Launch browser automatically
    - upload_size (str): Maximum file size for uploads
    - public (bool): Enable public URL mode
    - limit (int): Maximum number of concurrent requests
    """
    try:
        import gradio as gr

        system_theme = gr.themes.Default(
            primary_hue=gr.themes.colors.rose,
            secondary_hue=gr.themes.colors.rose,
            neutral_hue=gr.themes.colors.zinc
            )

        css = """
            ::-webkit-scrollbar {
                display: none;
            }

            ::-webkit-scrollbar-button {
                display: none;
            }

            footer {
                display: none !important;
            }

            body {
                -ms-overflow-style: none;
            }

            gradio-app {
                --body-background-fill: None;
            }

            footer {
                display: none !important;
            }

            .grid-wrap.svelte-hpz95u.svelte-hpz95u {
                overflow-y: auto;
            }
            
            input[type=range].svelte-pc1gm4 {
                background-image: linear-gradient(var(--color-accent), var(--color-accent));
            }
        """

        with gr.Blocks(css=css, title="Facemaker FM2", analytics_enabled=False, theme=system_theme, fill_height=True).queue(default_concurrency_limit=limit) as demo:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(f"## <br><center>Facemaker FM2 Web UI")
                    gr.Markdown(f"<center>Copyright (C) 2025 Ikmal Said. All rights reserved")
            
            return_output = gr.Checkbox(label="Return Output", value=True, visible=False)
            
            def update_paths(paths):
                return gr.Dropdown(choices=paths, value=paths[0] if paths else None)

            with gr.Tab("Image Face Swap"):
                with gr.Row():
                    with gr.Column():
                        target_image = gr.File(label="Target Image(s)", file_count="multiple", file_types=["image"])
                        target_image_list = gr.Dropdown(label="Target Image Paths", choices=['Please upload image(s)!'],
                                                        value='Please upload image(s)!')
                        target_preview = gr.Image(label="Target Preview")
                        source_image = gr.File(label="Source Image(s)", file_count="multiple", file_types=["image"])
                        source_preview = gr.Gallery(label="Source Preview", show_label=True, columns=3, height=240)
                        source_face_index = gr.Number(value=0, label="Source Face Index", precision=0)
                        target_face_index = gr.Number(value=0, label="Target Face Index", precision=0)
                        swap_all = gr.Checkbox(label="Swap All Faces", value=True)
                        detect_direction = gr.Dropdown(choices=['left-right', 'right-left', 'top-bottom', 'bottom-top', 
                                                       'small-large', 'large-small', 'best-worst', 'worst-best'],
                                                       value='left-right', label="Face Detection Direction")
                        img_swap_btn = gr.Button(variant="stop", value="Swap Faces")
                    with gr.Column():
                        output_image = gr.File(label="Result Image(s)", file_count="multiple", interactive=False)
                        output_image_list = gr.Dropdown(label="Result Image Paths", choices=['Please process the image first!'],
                                                        value='Please process the image first!')
                        output_preview = gr.Image(label="Result Preview")

                target_image.change(fn=update_paths, inputs=target_image, outputs=target_image_list)
                target_image_list.change(fn=lambda x: x, inputs=target_image_list, outputs=target_preview)
                source_image.change(fn=lambda x: x, inputs=source_image, outputs=source_preview)
                output_image_list.change(fn=lambda x: x, inputs=output_image_list, outputs=output_preview)

                img_swap_btn.click(fn=lambda *args: client.recognize_from_image(*args),
                                 inputs=[target_image, source_image, source_face_index, 
                                        target_face_index, swap_all, detect_direction, return_output],
                                 outputs=[output_image]).then(fn=update_paths, inputs=output_image,
                                                              outputs=[output_image_list])

            with gr.Tab("Video Face Swap"):
                with gr.Row():
                    with gr.Column():
                        target_video = gr.File(label="Target Video(s)", file_count="multiple", file_types=["video"])
                        target_video_list = gr.Dropdown(label="Target Video Paths", choices=['Please upload video(s)!'],
                                                        value='Please upload video(s)!')
                        target_video_preview = gr.Video(label="Target Video Preview")
                        source_image_vid = gr.File(label="Source Image(s)", file_count="multiple", file_types=["image"])
                        source_preview_vid = gr.Gallery(label="Source Preview", show_label=True, columns=3, height=240)
                        source_idx_vid = gr.Number(value=0, label="Source Face Index", precision=0)
                        target_idx_vid = gr.Number(value=0, label="Target Face Index", precision=0)
                        swap_all_vid = gr.Checkbox(label="Swap All Faces", value=True)
                        detect_direction_vid = gr.Dropdown(choices=['left-right', 'right-left', 'top-bottom', 'bottom-top',
                                                           'small-large', 'large-small', 'best-worst', 'worst-best'],
                                                           value='left-right', label="Face Detection Direction")
                        vid_swap_btn = gr.Button(variant="stop", value="Process Video")
                    with gr.Column():
                        output_video = gr.File(label="Result Video", file_count="multiple", interactive=False)
                        output_video_list = gr.Dropdown(label="Result Video Paths", choices=['Please process the video first!'],
                                                        value='Please process the video first!')
                        output_video_preview = gr.Video(label="Result Preview")

                target_video.change(fn=update_paths, inputs=target_video, outputs=target_video_list)
                target_video_list.change(fn=lambda x: x, inputs=target_video_list, outputs=target_video_preview)
                source_image_vid.change(fn=lambda x: x, inputs=source_image_vid, outputs=source_preview_vid)
                output_video_list.change(fn=lambda x: x, inputs=output_video_list, outputs=output_video_preview)

                vid_swap_btn.click(fn=lambda *args: client.recognize_from_video(*args),
                                 inputs=[target_video, source_image_vid, source_idx_vid,
                                        target_idx_vid, swap_all_vid, detect_direction_vid, return_output],
                                 outputs=[output_video]).then(fn=update_paths, inputs=output_video, 
                                                            outputs=[output_video_list])

            with gr.Tab("Image Face Detection"):
                with gr.Row():
                    with gr.Column():
                        detect_image = gr.Image(label="Input Image", type="filepath")
                        return_thumbnails = gr.Checkbox(label="Return Face Thumbnails", value=True)
                        return_src = gr.Checkbox(label="Return Marked Image", value=True) 
                        detect_direction = gr.Dropdown(choices=['left-right', 'right-left', 'top-bottom', 'bottom-top',
                                                              'small-large', 'large-small', 'best-worst', 'worst-best'],
                                                               value='left-right', label="Face Detection Direction")
                        detect_btn = gr.Button(variant="stop", value="Detect Faces")
                    with gr.Column():
                        src_image = gr.Image(label="Marked Image")
                        face_thumbnails = gr.Gallery(label="Face Thumbnails", columns=3, height=240)
                        total_faces = gr.Label(label="Total Faces Found")
                        faces_info = gr.JSON(label="Detected Faces Info")

                detect_btn.click(fn=lambda *args: client.get_faces_from_image(*args),
                               inputs=[detect_image, return_thumbnails, return_src, detect_direction],
                               outputs=[faces_info, total_faces, face_thumbnails, src_image])

            with gr.Tab("Video Face Detection"):
                with gr.Row():
                    with gr.Column():
                        detect_video = gr.Video(label="Input Video")
                        return_thumbnails_vid = gr.Checkbox(label="Return Face Thumbnails", value=True)
                        return_src_vid = gr.Checkbox(label="Return Marked Video", value=True)
                        detect_direction_vid = gr.Dropdown(choices=['left-right', 'right-left', 'top-bottom', 'bottom-top',
                                                              'small-large', 'large-small', 'best-worst', 'worst-best'],
                                                               value='left-right', label="Face Detection Direction")
                        detect_btn_vid = gr.Button(variant="stop", value="Detect Faces")
                    with gr.Column():
                        src_video = gr.Video(label="Marked Video")
                        face_thumbnails_vid = gr.Gallery(label="Face Thumbnails", columns=3, height=240)
                        total_faces_vid = gr.Label(label="Total Faces Found")
                        faces_info_vid = gr.JSON(label="Detected Faces Info")

                detect_btn_vid.click(fn=lambda *args: client.get_faces_from_video(*args),
                                   inputs=[detect_video, return_thumbnails_vid, return_src_vid, detect_direction_vid],
                                   outputs=[faces_info_vid, total_faces_vid, face_thumbnails_vid, src_video])

            with gr.Tab("Image Face Enhancement"):
                with gr.Row():
                    with gr.Column():
                        target_image_enh = gr.File(label="Target Image(s)", file_count="multiple", file_types=["image"])
                        target_image_list_enh = gr.Dropdown(label="Target Image Paths", choices=['Please upload image(s)!'],
                                                            value='Please upload image(s)!')
                        target_preview_enh = gr.Image(label="Target Preview")
                        target_face_index_enh = gr.Number(value=0, label="Target Face Index", precision=0)
                        enhance_all = gr.Checkbox(label="Enhance All Faces", value=True)
                        detect_direction_enh = gr.Dropdown(choices=['left-right', 'right-left', 'top-bottom', 'bottom-top',
                                                              'small-large', 'large-small', 'best-worst', 'worst-best'],
                                                               value='left-right', label="Face Detection Direction")
                        enhance_btn = gr.Button(variant="stop", value="Enhance Faces")
                    with gr.Column():
                        output_image_enh = gr.File(label="Result Image(s)", file_count="multiple", interactive=False)
                        output_image_list_enh = gr.Dropdown(label="Result Image Paths", choices=['Please process the image first!'],
                                                        value='Please process the image first!')
                        output_preview_enh = gr.Image(label="Result Preview")

                target_image_enh.change(fn=update_paths, inputs=target_image_enh, outputs=target_image_list_enh)
                target_image_list_enh.change(fn=lambda x: x, inputs=target_image_list_enh, outputs=target_preview_enh)
                output_image_list_enh.change(fn=lambda x: x, inputs=output_image_list_enh, outputs=output_preview_enh)

                enhance_btn.click(fn=lambda *args: client.enhance_from_image(*args),
                                inputs=[target_image_enh, target_face_index_enh, 
                                       enhance_all, detect_direction_enh, return_output],
                                outputs=[output_image_enh]).then(fn=update_paths, inputs=output_image_enh,
                                                                 outputs=[output_image_list_enh])

            with gr.Tab("Video Face Enhancement"):
                with gr.Row():
                    with gr.Column():
                        target_video_enh = gr.File(label="Target Video(s)", file_count="multiple", file_types=["video"])
                        target_video_list_enh = gr.Dropdown(label="Target Video Paths", choices=['Please upload video(s)!'],
                                                            value='Please upload video(s)!')
                        target_preview_enh = gr.Video(label="Target Preview")
                        target_face_index_enh = gr.Number(value=0, label="Target Face Index", precision=0)
                        enhance_all = gr.Checkbox(label="Enhance All Faces", value=True)
                        detect_direction_enh = gr.Dropdown(choices=['left-right', 'right-left', 'top-bottom', 'bottom-top',
                                                              'small-large', 'large-small', 'best-worst', 'worst-best'],
                                                               value='left-right', label="Face Detection Direction")
                        enhance_btn = gr.Button(variant="stop", value="Enhance Faces")
                    with gr.Column():
                        output_video_enh = gr.File(label="Result Video(s)", file_count="multiple", interactive=False)
                        output_video_list_enh = gr.Dropdown(label="Result Video Paths", choices=['Please process the video first!'],
                                                        value='Please process the video first!')
                        output_preview_enh = gr.Video(label="Result Preview")

                target_video_enh.change(fn=update_paths, inputs=target_video_enh, outputs=target_video_list_enh)
                target_video_list_enh.change(fn=lambda x: x, inputs=target_video_list_enh, outputs=target_preview_enh)
                output_video_list_enh.change(fn=lambda x: x, inputs=output_video_list_enh, outputs=output_preview_enh)

                enhance_btn.click(fn=lambda *args: client.enhance_from_video(*args),
                                inputs=[target_video_enh, target_face_index_enh, 
                                       enhance_all, detect_direction_enh, return_output],
                                outputs=[output_video_enh]).then(fn=update_paths, inputs=output_video_enh,
                                                                 outputs=[output_video_list_enh])

            with gr.Tab("Settings"):
                face_swapper_precision = gr.Dropdown(choices=list(client.fm_models['face_swapper'].keys()),
                                                    value=client.face_swapper_precision, label="Face Swapper Precision")
                face_landmarker = gr.Dropdown(choices=list(client.fm_models['face_landmarker'].keys()),
                                            value=client.face_landmarker, label="Face Landmarker Model")
                face_enhancer = gr.Dropdown(choices=list(client.fm_models['face_enhancer'].keys()),
                                          value=client.face_enhancer, label="Face Enhancer Model")
                skip_enhancer = gr.Checkbox(label="Skip Face Enhancer", value=client.skip_enhancer)
                detector_score = gr.Slider(0, 1, value=client.face_detector_score, label="Face Detector Score")
                ref_distance = gr.Slider(0, 1, value=client.reference_face_distance, label="Reference Face Distance")
                save_format = gr.Dropdown(choices=['webp', 'jpg', 'png'],
                                         value=client.save_format, label="Save Format")
                save_to = gr.Textbox(label="Save Location", value=str(client.save_to))

                update_btn = gr.Button(variant="stop", value="Update Settings")

                update_btn.click(fn=client.change_settings,
                               inputs=[face_swapper_precision, face_landmarker, face_enhancer, detector_score, ref_distance, save_format, save_to, skip_enhancer])

        demo.launch(
            server_name=host,
            server_port=port,
            inbrowser=browser,
            max_file_size=upload_size,
            share=public,
            quiet=True
        )
        
    except Exception as e:
        client.logger.error(f"{str(e)}")
        raise