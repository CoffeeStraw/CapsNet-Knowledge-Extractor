/**
 * @fileOverview Main Workflow for page loading and setup.
 * @author Antonio Strippoli
 */
import { TrainingStepSelector, ModelSelector } from './core/selectors.js'
import { visualizeComputeStep, visualizeLayersTabs } from './core/visualizers.js'

async function main() {
    // Request models' structure from API
    var response = await Promise.resolve($.get("/api/getModels"))

    if (response['status'] != 200)
        alert("Error: Backend reported an error with status " + response['status'])

    var models = response['models']

    // Build model selector based on the models' name
    var ms = new ModelSelector(Object.keys(models))
    // Build a placeholder training step slider
    var tss = new TrainingStepSelector(['1-0', '2-0', '3-0', 'trained'])
    tss.disable()

    // Set listener for model selector changes
    ms.selector.on("changed.bs.select", function () {
        // Enable forms for input if first time
        if (ms.selected == 'Unpicked') {
            // 1st form
            $('#dataset-i').removeAttr('disabled')
            $('#dataset-i-submit').removeAttr('disabled')
            // 2nd form
            $('#custom-image').removeAttr('disabled')
            $('#custom-image-label').removeClass('disabled')
            // 3rd form
            $('#rotation-value').removeAttr('disabled')
            $('#invert-colors').removeAttr('disabled')
            // General submit button
            $('#forms-general-submit').removeAttr('disabled')
            // Training Step Selector
            tss.enable()
        }

        ms.selected = this.value

        // Update training step selector
        tss.changeElements(models[ms.selected]['training_steps'])

        // Visualize tabs for the current model
        visualizeLayersTabs(models[ms.selected]['layers'])
    });

    // Set listener for training step selector changes
    tss.slider.noUiSlider.on('set.one', function (values, handle) {
        tss.selected = tss.elements[Math.round(values[handle])]
    });

    // Set listeners for inputs cleanup and preview
    document.getElementById('dataset-i').addEventListener('change', function (event) {
        // Clear
        document.getElementById('custom-image').value = "";
        // Update preview
        // TODO: set preview based on changes (require backend)
    })
    document.getElementById('custom-image').addEventListener('change', function (event) {
        // Clear
        document.getElementById('dataset-i').value = ""
        // Update preview
        var val = document.getElementById('custom-image').files[0]
        // TODO: set preview based on changes (require backend)
    })

    // Function to execute on every form's submit
    async function sendForVisualization(event) {
        // Prevent default page reload
        event.preventDefault()

        // START TO COLLECT DATA
        let form_data = new FormData();

        form_data.append("model", ms.selected)
        form_data.append("step", tss.selected)

        // === GET Transformations ===
        var rot_value = document.getElementById('rotation-value').value
        var invert_colors = document.getElementById('invert-colors').checked

        form_data.append("rotation", rot_value != "" ? parseInt(rot_value) : 0)
        form_data.append("invert_colors", invert_colors)

        // === Image input source ===
        // Note: one of the two input for the image will always be void.
        // This way, we can determine which choice the user has made.
        var testset_index = document.getElementById('dataset-i').value
        if (testset_index != "")
            form_data.append("testset_index", parseInt(testset_index))
        else {
            var custom_image_obj = document.getElementById('custom-image')
            if (custom_image_obj.value === "")
                return alert("Either enter a test set index or send a custom image.")
            form_data.append("image", custom_image_obj.files[0]);
        }

        // Request to the API to compute the outputs
        $.ajax({
            url: '/api/computeStep',
            cache: false,
            contentType: false,
            processData: false,
            data: form_data,
            type: 'post',
            success: function (response) {
                // Visualize input image and outputs in the tabs
                visualizeComputeStep(response)
            },
            error: function (response) {
                alert("ERROR: " + response["responseJSON"]["error_description"]);
            }
        });
    }

    // Set listeners for forms' submits
    document.getElementById('form-image-selection').addEventListener('submit', sendForVisualization)
    document.getElementById('form-transformations').addEventListener('submit', sendForVisualization)
    document.getElementById('forms-general-submit').addEventListener('click', sendForVisualization)
}

main()