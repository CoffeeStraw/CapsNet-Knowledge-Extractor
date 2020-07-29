/**
 * @fileOverview Main Workflow for page loading.
 * @author Antonio Strippoli
 */
import { TrainingStepSelector, ModelSelector } from './core/selectors.js'
import { visualizeComputeStep, visualizeLayersTabs } from './core/visualizer.js'

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
            // Enable forms for input if first time
            if (ms.selected == 'Unpicked') {
                // 1st form
                $('#dataset-i').removeAttr('disabled')
                $('#dataset-i-submit').removeAttr('disabled')
                // 2nd form
                $('#custom-image').removeAttr('disabled')
                $('#custom-image-label').removeClass('disabled')
                // Training Step Selector
                tss.enable()
            }

            ms.selected = this.value

            // Update training step selector
            tss.changeElements(models[ms.selected]['training_steps'])

            // Visualize tabs for the current model
            visualizeLayersTabs(models[ms.selected]['layers'])
        }
    );

    // Set listener for training step selector changes
    tss.slider.noUiSlider.on('set.one', function (values, handle) {
        tss.selected = tss.elements[Math.round(values[handle])]
    });

    // Set listeners for forms' submits
    document.getElementById('form-dataset').addEventListener('submit', async function (event) {
        event.preventDefault()
        var datasetIndex = parseInt(document.getElementById('dataset-i').value)

        // Everything under this comment should go in another file
        // Request to the API to compute the outputs (TODO: different datasets support)
        var response = await Promise.resolve(
            $.post({
                url: "/api/computeStep",
                contentType: "application/json",
                data: JSON.stringify({
                    model: ms.selected,
                    step: tss.selected,
                    dataset: "MNIST",
                    index: datasetIndex
                }),
            })
        )

        // Visualize input image and outputs in the tabs
        visualizeComputeStep(response)
    })
}

main()