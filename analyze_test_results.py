import fire 
import pandas as pd 
import glob
import os 
 
def main(
    excel_dir: str="",
    save_raw_df: bool=True,
):
    
    # get the file path 
    benchmark = glob.glob(os.path.join(excel_dir, 'benchmark.xlsx'))
    extended = glob.glob(os.path.join(excel_dir, 'extended.xlsx'))
    fsdp = glob.glob(os.path.join(excel_dir, 'fsdp.xlsx'))
    generation = glob.glob(os.path.join(excel_dir, 'generation.xlsx'))
    peft = glob.glob(os.path.join(excel_dir, 'peft_integration.xlsx'))
    quantization = glob.glob(os.path.join(excel_dir, 'quantization.xlsx'))
    trainer = glob.glob(os.path.join(excel_dir, 'trainer.xlsx'))
    pipelines = glob.glob(os.path.join(excel_dir, 'pipelines.xlsx'))
    singles = glob.glob(os.path.join(excel_dir, 'single_files.xlsx'))
    models = glob.glob(os.path.join(excel_dir, '*_models.xlsx'))
    
    # read in the file  
    benchmark_df = pd.read_excel(benchmark[0])
    extended_df = pd.read_excel(extended[0])
    fsdp_df = pd.read_excel(fsdp[0])
    generation_df = pd.read_excel(generation[0])
    peft_df = pd.read_excel(peft[0])
    quantization_df = pd.read_excel(quantization[0])
    trainer_df = pd.read_excel(trainer[0])
    pipelines_df = pd.read_excel(pipelines[0])
    singles_df = pd.read_excel(singles[0])
    models_df = pd.concat(pd.read_excel(excel_file) for excel_file in models).reset_index()
    all_df = pd.concat([benchmark_df, extended_df, fsdp_df, generation_df, peft_df, quantization_df, trainer_df, pipelines_df, singles_df, models_df]).reset_index()
    all_df.drop(columns=['level_0', 'index'], inplace=True)
    
    models_df = models_df[['file_name', 'suite_name', 'test_name',  'result', 'message', 'duration']]
    all_df = all_df[['file_name', 'suite_name', 'test_name',  'result', 'message', 'duration']]
    
    def replace_unittests(df):
        for i, v in df['file_name'].items():
            if '/unittest/' in v:
                suite_name = df.loc[i, 'suite_name']
                new_df = df.groupby(['suite_name', 'file_name'], as_index=False).size().sort_values(['size'], ascending=False)
                new_value = new_df[new_df['suite_name'] == suite_name]['file_name'].iloc[0]
                df.loc[i, 'file_name'] = new_value
        return df 

    all_df = replace_unittests(all_df)
    models_df = replace_unittests(models_df)

    if save_raw_df: 
        models_df.to_excel("df_raw_models.xlsx", index=False)
        all_df.to_excel("df_raw_all.xlsx", index=False)
    
    all_agg = all_df.groupby(['file_name','result'], as_index=False).size()
    
    def transform_df(df, col_name):
        values = df[col_name].unique().tolist()
        passed = []
        skipped = []
        failed = []
        error = []
        for i, value in enumerate(values):
            tmp_df = df[df[col_name] == value]
            passed.append(0)
            skipped.append(0)
            failed.append(0)
            error.append(0)
            for i, v in tmp_df['result'].items():
                if v == 'PASSED':
                    passed[-1] = tmp_df['size'][i] 
                elif v == 'SKIPPED':
                    skipped[-1] = tmp_df['size'][i]
                elif v == 'ERROR':
                    error[-1] = tmp_df['size'][i]
                else:
                    failed[-1] = tmp_df['size'][i]
                
        new_df = pd.DataFrame({'Test Vector': values, 'PASS': passed, 'FAIL': failed, 'SKIP': skipped, 'ERROR': error})
        new_df['SKIP'] = new_df['SKIP'] + new_df['ERROR']
        new_df['TOTAL'] = new_df['PASS'] + new_df['FAIL'] + new_df['SKIP']
        new_df.drop(columns=['ERROR'], inplace=True)
        
        return new_df 
    
    all_df_tran = transform_df(all_agg, 'file_name')
    
    def extract_category(all_df):
        category = []
        for i, v in all_df['Test Vector'].items():
            end = v.split('/')
            if len(end) == 2:
                category.append(end[-1])
            else:
                category.append(end[1])     
        all_df['Category'] = category    
        return all_df 
    
    all_df_tran = extract_category(all_df_tran)

    all_df_tran = all_df_tran[['Category', 'Test Vector', 'PASS', 'FAIL', 'SKIP', 'TOTAL']]
    all_df_tran.to_excel("df_stats.xlsx", index=False)
    all_df_tran2 = all_df_tran.groupby(['Category']).agg({'PASS':'sum', 'FAIL': 'sum', 'SKIP': 'sum', 'TOTAL': 'sum'}).reset_index()  
    all_df_tran2.to_excel("df_stats_v2.xlsx", index=False)
    
    models_skip_stats = models_df[models_df['result'] == 'SKIPPED']['message'].value_counts().reset_index()
    models_skip_stats.to_excel("df_models_skipped_stats.xlsx", index=False)
    
    
if __name__ == '__main__':
    fire.Fire(main)