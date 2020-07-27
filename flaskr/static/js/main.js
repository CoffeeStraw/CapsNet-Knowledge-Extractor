/*
Main Workflow for page loading
Author: Antonio Strippoli
*/
import { TrainingStepSelector, ModelSelector } from './core/selectors.js'

async function main() {
    // Request models' structure from API
    var response = await Promise.resolve($.get("/api/getModels"))

    if (response['status'] != 200)
        return 'Error' // TODO: better error exception

    var models = response['models']

    // Build model selector based on the models' name
    var ms = new ModelSelector(Object.keys(models))
    // Build a placeholder training step slider
    var tss = new TrainingStepSelector(['1-0', '2-0', '3-0', 'trained'])
    tss.disable()

    // Set listener for model selector changes
    ms.selector.on("changed.bs.select",
        function () {
            var model_name = this.value

            // Update
            tss.changeElements(models[model_name]['training_steps'])
            tss.enable()
        });
}

main()