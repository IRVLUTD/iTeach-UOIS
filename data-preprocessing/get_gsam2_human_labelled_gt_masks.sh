# This is a human aided labelling script to generate ground truth masks for the iTeach-UOIS dataset.
# NOTE: This is only for sanity check, nowhere in the results these have been used.
# With this experiment it was clear that
# - For some cases, the model is able to detect objects in the scene using GSAM2 pipeline
# - For some cases, the model is not able to detect objects in the scene using GSAM2 pipeline
    # - Here, manual bbox prompts were given
# - Hence research on unseen object detection is still an open question

################################################################## today

python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene1/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene3/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene6/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene7/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene16/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene22/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene24/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene25/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene30/jpg


#  changed the alpha for mask overlay
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene2/jpg --text_prompt "objects on tabletop" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene4/jpg --text_prompt "objects"  --n 2
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene5/jpg --text_prompt "objects on couch" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene9/jpg --text_prompt "objects on trash" --n 3
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene10/jpg --text_prompt "objects on rack" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene11/jpg --text_prompt "objects on couch" --n 3
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene12/jpg --text_prompt "objects on black floor" --n 0

python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene13/jpg --text_prompt "kitchen objects without shadow" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene14/jpg --text_prompt "objects" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene15/jpg --text_prompt "objects" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene17/jpg --text_prompt "objects" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene18/jpg --text_prompt "objects" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene19/jpg --text_prompt "objects" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene20/jpg --text_prompt "objects" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene21/jpg --text_prompt "objects" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene23/jpg --text_prompt "daily objects" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene27/jpg --text_prompt "objects on shelf" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene28/jpg --text_prompt "objects" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene29/jpg --text_prompt "objects" --n 0
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene31/jpg --text_prompt "objects on couch" --n 2
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene32/jpg --text_prompt "objects" --n 1
python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene33/jpg --text_prompt "objects" --n 1


for i in {34..50}; do python test_gdino_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene$i/jpg --text_prompt "objects" --n 1; done

python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene34/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene38/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene42/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene43/jpg
python test_bbox_prompt_samv2.py --input_dir ~/iTeach-UOIS-Data-Collection/data/scene44/jpg
