/**
 * @fileOverview Main Workflow for page loading.
 * @author Antonio Strippoli
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

            // Prepare visualization section for the selected model
            var visualization_html = ''
            var visualization_content_html = ''

            models[ms.selected]['layers'].forEach(function (value, i) {
                var [name, type] = value
                if (i == 0) {
                    visualization_content_html += `
                    <div class="tab-pane fade show active" id="${name}" role="tabpanel" aria-labelledby="${name}-tab">Send an image!</div>`

                    visualization_html += `
                    <li class="nav-item">
                            <a class="nav-link active show" id="${name}-tab" data-toggle="tab" href="#${name}"
                            role="tab" aria-controls="${name}" aria-selected="true">${name}</a>
                    </li>`
                } else {
                    visualization_content_html += `
                    <div class="tab-pane fade" id="${name}" role="tabpanel" aria-labelledby="${name}-tab">Send an image!</div>`

                    visualization_html += `
                    <li class="nav-item">
                        <a class="nav-link"  id="${name}-tab" data-toggle="tab" href="#${name}"
                        role="tab" aria-controls="${name}" aria-selected="false">${name}</a>
                    </li>`
                }
            })

            // Update htmls
            $('#visualization').html(visualization_html)
            $('#visualization-content').html(visualization_content_html)
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
        // Request to the API to compute the outputs
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

        // Replace input image with the choosen one
        document.getElementById("input-img").src = response['out_dir'] + "img.jpeg"

        // Visualize every image produced

        for (var layer_name in response['layers_outs']) {
            var layer_out_dir = response['out_dir'] + `${layer_name}/`
            var visualization_content_html = ''

            response['layers_outs'][layer_name].forEach(function (img_name) {
                var img_path = layer_out_dir + img_name
                visualization_content_html += `
                <img class="img-responsive img-gallery" src="${img_path}">
                `
            })

            $(`#${layer_name}`).html(visualization_content_html)
        }
    })
}

main()