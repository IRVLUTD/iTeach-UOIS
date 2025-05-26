## Prepare data for SAM2
Convert the pngs to jpgs and rename in reverse manner
```python
python convert2jpg_in_reverse.py --input_dir /home/jishnu/iTeach-UOIS-Data-Collection/data/training_set/042
2T130100_dummy
```

### Propogate mask in reverse
```python
python propogate_masks_via_bbox_prompt_samv2.py --input_dir /home/jishnu/iTeach-UOIS-Data-Collection/data/t
raining_set/0422T130100_dummy/
```

### Test model ckpt
(msm39) root@IRVL-001:/home/jishnu/Projects/iTeach-UOIS/uois-models/UnseenObjectsWithMeanShift/lib/fcn# python iteach_test_dataset.py test_new_hl
